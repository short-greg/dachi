"""
Defines core Msg, Prompt, Resp, DeltaResp and Dialog classes for conversation handling.

This module provides a simplified message architecture where:
- Msg: Base message class with core content and metadata
- Prompt(Msg): User prompts with LLM configuration and sampling parameters  
- Resp(Msg): LLM responses with generation metadata and tool execution
- DeltaResp: Streaming deltas containing only incremental changes

The architecture eliminates the complex spawn() logic and streaming state management
of the previous implementation, providing cleaner inheritance and simpler usage.
"""

# 1st party
from __future__ import annotations
import typing
import typing as t
from abc import abstractmethod
from typing import Self, Literal
import base64
from datetime import datetime, timezone

# 3rd party
from pydantic import BaseModel, PrivateAttr, Field
import pydantic

# local
from . import Renderable
from ._tool import ToolUse, BaseTool

try:
    # pydantic v2 preferred
    from pydantic import BaseModel, Field
    _PD_V2 = True
except Exception:  # pragma: no cover
    # pydantic v1 fallback
    from pydantic import BaseModel, Field  # type: ignore
    _PD_V2 = False


class _Final:
    """A unique object to mark the end of a streaming response."""
    def __repr__(self):
        return "<Final Token>"

END_TOK = _Final()
NULL_TOK = object()


class Attachment(BaseModel):
    """
    Declarative reference to a non-text asset.

    Attributes:
        kind: 'image' | 'audio' | 'video' | 'file' | 'data'
        data: The raw data in bytes or the url for the data.
        mime: Optional MIME type, e.g. 'image/png'.
        name: Human-friendly label/filename.
        purpose: Hint for adapters, e.g. 'vision_input', 'context_doc'.
        info: Small, inspectable metadata. Large payloads should be external.
    """
    kind: t.Literal['image', 'audio', 'video', 'file', 'data']
    data: str = Field(
        description="The raw data in bytes or the url for the data."
    )
    mime: t.Optional[str] = Field(default=None, description="MIME type of the attachment, e.g. 'image/png'.")
    name: t.Optional[str] = Field(default=None, description="Human-friendly label/filename.")
    purpose: t.Optional[str] = Field(default=None, description="Hint for adapters, e.g. 'vision_input', 'context_doc'.")
    info: t.Dict[str, t.Any] = Field(default_factory=dict)


def to_b64(filepath) -> str:
    """ Convert a file to a base64 string.
    """
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Msg(BaseModel):
    """
    Base message class representing a complete message in a conversation.
    
    This class contains the core content and metadata for any message type.
    Subclasses like Prompt and Resp add specialized attributes for their use cases.
    
    Key concepts:
        - Role: Indicates who sent the message ('user', 'assistant', 'system', etc.)
        - Text: The message content, can be plain text or structured dict
        - Tool calls: Completed tool/function calls with their results
        - Attachments: Files, images, or other media associated with the message
        - Metadata: Extensible storage for additional information
        
    Usage:
        msg = Msg(role='user', text='Hello, how are you?')
        msg = Msg(role='assistant', text={'final': 'The answer is 42', 'thinking': 'Let me calculate...'})
    """
    # Core message content
    role: str
    alias: t.Optional[str] = None
    text: t.Optional[str] = Field(
        default=None, 
        description="Text content as string."
    )
    
    # Rich content  
    attachments: t.List[Attachment] = Field(
        default_factory=list, 
        description="List of attachments included in the message"
    )
    tool_calls: t.List[ToolUse] = Field(
        default_factory=list, 
        description="List of completed tool calls with their results stored in ToolUse.result"
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        description="Timestamp of the message creation in UTC"
    )
    tags: t.List[str] = Field(default_factory=list, description="List of tags associated with the message")
    meta: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description="Metadata dictionary for additional information"
    )

    model_config = pydantic.ConfigDict(extra="allow")

    def render(self) -> str:
        """Compact, human-readable line for logs."""
        name = self.alias or self.role
        if isinstance(self.text, str):
            text = self.text
        elif isinstance(self.text, dict):
            text = self.text.get("final") or self.text.get("text") or ""
        else:
            text = ""
        extras: t.List[str] = []
        if self.attachments:
            extras.append(f"attachments={len(self.attachments)}")
        if self.tool_calls:
            extras.append(f"tool_calls={len(self.tool_calls)}")
        suffix = f"  [{' | '.join(extras)}]" if extras else ""
        return f"{name}: {text}{suffix}"

    def clone(self) -> 'Msg':
        """Create a copy of this message.

        Returns:
            Msg: A new message instance with the same data
        """
        return self.model_copy(deep=True)


class Prompt(Msg):
    """
    User prompt message with LLM configuration and sampling parameters.
    
    Extends the base Msg with prompt-specific settings that control LLM behavior,
    tool availability, output formatting, and generation parameters.
    
    Key features:
        - Tool configuration: Override available tools for this prompt
        - Schema control: Force specific output formats (JSON, text, structured)
        - Sampling parameters: Control temperature, max_tokens, etc.
        - Advanced features: System prompt overrides, reasoning controls
        
    Usage:
        prompt = Prompt(
            text="Analyze this data",
            tools=[my_analysis_tool],
            temperature=0.7,
            max_tokens=1000,
            format_override=MyOutputModel
        )
    """
    role: str = "user"  # Default for prompts
    
    # LLM Sampling parameters (commonly passed via **kwargs)
    model: t.Optional[str] = Field(default=None, description="Model override")
    temperature: t.Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: t.Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: t.Optional[float] = Field(default=None, description="Nucleus sampling")
    frequency_penalty: t.Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: t.Optional[float] = Field(default=None, description="Presence penalty")
    seed: t.Optional[int] = Field(default=None, description="Deterministic seed")
    
    # Advanced prompt features
    system_message: t.Optional[str] = Field(default=None, description="System message override")
    reasoning_summary_request: t.Optional[bool] = Field(
        default=None, 
        description="Request reasoning summary for reasoning models"
    )
    # Tool configuration
    tool_override: bool = Field(
        default=False, 
        description="Defines whether tools should override those in previous prompts"
    )
    tools: t.Optional[t.List[BaseTool]] = Field(
        default=None, 
        description="Available tools for LLM. If None, uses sequence priority"
    )
    
    # Output format control
    format_override: t.Optional[t.Union[Literal["json", "text"], t.Type[pydantic.BaseModel], dict]] = Field(
        default=None,
        description="Override output format: 'json'/'text', Pydantic model class, or JSON schema dict for structured output"
    )


class Resp(Msg):
    """
    LLM response message with generation metadata and tool execution capabilities.
    
    Extends the base Msg with response-specific metadata from LLM generation,
    including usage statistics, reasoning content, tool execution, and processing results.
    
    Key features:
        - Generation metadata: Model, finish reason, token usage
        - Advanced features: Reasoning content, citations, log probabilities
        - Tool execution: Tools to be executed (tool_use) vs completed (tool_calls)
        - Processing output: Results from ToOut processors
        - Raw data: Access to original LLM response for debugging
        
    Usage:
        resp = Resp(
            text="The analysis shows...",
            model="gpt-4",
            usage={"prompt_tokens": 100, "completion_tokens": 200},
            tool_use=[tool_to_execute]
        )
    """
    role: str = "assistant"  # Default for responses
    
    # LLM Response metadata  
    model: t.Optional[str] = Field(default=None, description="Model that generated response")
    finish_reason: t.Optional[str] = Field(
        default=None, 
        description="Reason generation stopped (e.g., 'stop', 'length', 'tool_calls')"
    )
    
    # Usage/billing information
    usage: t.Dict[str, t.Any] = Field(
        default_factory=dict, 
        description="Token usage statistics (prompt_tokens, completion_tokens, etc.)"
    )
    
    # Advanced LLM features
    logprobs: t.Optional[t.Dict] = Field(
        default=None, 
        description="Log probabilities for generated tokens"
    )
    thinking: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = Field(
        default=None, 
        description="Reasoning content for reasoning models (o1, etc.)"
    )
    citations: t.Optional[t.List[t.Dict]] = Field(
        default=None, 
        description="Source citations for generated content"
    )
    # Tool execution
    tool_use: t.List[ToolUse] = Field(
        default_factory=list, 
        description="Tools to be executed (not yet completed)"
    )
    # Message metadata
    id: t.Optional[str] = Field(default=None, description="Unique identifier for the message")
    # Processing output
    out: t.Any = Field(
        default=None, 
        description="Processed result from ToOut processors"
    )
    
    # Multi-choice support
    choices: t.Optional[t.List[t.Dict[str, t.Any]]] = Field(
        default=None, 
        description="Alternative completions for multi-choice scenarios"
    )
    

    
    # Internal
    _raw: t.Dict = PrivateAttr(default_factory=dict)
    
    @property
    def raw(self) -> t.Dict:
        """Get the raw data from the LLM response."""
        return self._raw
        
    @raw.setter  
    def raw(self, value: t.Dict):
        """Set the raw data for the LLM response."""
        self._raw = value
        
    def use_tool(self, idx: t.Optional[int] = None):
        """Execute the tool calls in the response and return the results.
        
        Args:
            idx: Index of specific tool to execute, or None to execute all
            
        Returns:
            Dict[str, Any]: Dictionary mapping tool names to their results
        """
        result = {}
        tools_to_execute = []
        
        if idx is None:
            tools_to_execute = self.tool_use[:]  # Copy all tools
        else:
            if 0 <= idx < len(self.tool_use):
                tools_to_execute = [self.tool_use[idx]]
        
        for tool_use in tools_to_execute:
            res = tool_use()  # Execute the tool
            result[tool_use.option.name] = res
            
            # Move executed tool from tool_use to tool_calls
            if tool_use in self.tool_use:
                self.tool_use.remove(tool_use)
                self.tool_calls.append(tool_use)
        
        return result


class DeltaResp(BaseModel):
    """
    Streaming delta information containing only incremental changes.
    
    This class represents a single streaming chunk with only the new content
    that was added in this specific chunk. It does NOT contain accumulated values.
    
    Field Meanings:
        text: New text content in this chunk only
        thinking: New reasoning content in this chunk only
        citations: New citation information in this chunk only
        tool: Partial tool call JSON fragment for this chunk  
        finish_reason: Set only when streaming completes
        usage: Token usage for this specific chunk
        
    Important: These are chunk deltas, not accumulated values.
    For accumulated values, use the parent Resp object fields.
    """
    
    text: t.Optional[str] = Field(
        default=None, 
        description="Incremental text content for this streaming chunk only"
    )
    thinking: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = Field(
        default=None, 
        description="Incremental reasoning content for this chunk only"
    )
    citations: t.Optional[t.List[t.Dict]] = Field(
        default=None, 
        description="Incremental citation information for this chunk only"
    )
    tool: t.Optional[str] = Field(
        default=None, 
        description="Partial tool call arguments (JSON fragment) for this chunk"
    )
    finish_reason: t.Optional[str] = Field(
        default=None, 
        description="Reason generation stopped - set only when streaming completes"
    )
    usage: t.Optional[t.Dict[str, t.Any]] = Field(
        default=None, 
        description="Per-chunk token usage statistics for this specific chunk"
    )
    accumulation: t.Dict[str, t.Any] = Field(
        default_factory=dict,
        description="Accumulated values up to this chunk"
    )

