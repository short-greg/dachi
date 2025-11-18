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
    text: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = Field(
        default="", 
        description="Text content as string or structured dict with channels (e.g., final, thinking)"
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
    
    # Message metadata
    id: t.Optional[str] = Field(default=None, description="Unique identifier for the message")
    prev_id: t.Optional[str] = Field(default=None, description="ID of the previous message in the conversation")
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
    
    # LLM Sampling parameters (commonly passed via **kwargs)
    model: t.Optional[str] = Field(default=None, description="Model override")
    temperature: t.Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: t.Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: t.Optional[float] = Field(default=None, description="Nucleus sampling")
    frequency_penalty: t.Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: t.Optional[float] = Field(default=None, description="Presence penalty")
    seed: t.Optional[int] = Field(default=None, description="Deterministic seed")
    
    # Advanced prompt features
    system_prompt: t.Optional[str] = Field(default=None, description="System message override")
    reasoning_summary_request: t.Optional[bool] = Field(
        default=None, 
        description="Request reasoning summary for reasoning models"
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
    
    # Multi-choice support
    choices: t.Optional[t.List[t.Dict[str, t.Any]]] = Field(
        default=None, 
        description="Alternative completions for multi-choice scenarios"
    )
    
    # Tool execution
    tool_use: t.List[ToolUse] = Field(
        default_factory=list, 
        description="Tools to be executed (not yet completed)"
    )
    
    # Processing output
    out: t.Any = Field(
        default=None, 
        description="Processed result from ToOut processors"
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


class BaseDialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """
    _renderer: typing.Callable[[typing.List[Msg]], str] = pydantic.PrivateAttr(
        default=None
    )

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        pass

    @abstractmethod
    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        pass

    @abstractmethod
    def pop(self, index: int=-1) -> Msg:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        pass

    @abstractmethod
    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        pass

    def append(self, message: Msg) -> 'BaseDialog':
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        return self.insert(
            message=message, ind=None
        )

    def add(self, message: t.Union[Msg, Resp]) -> Msg:
        """Alias for append to add a message to the end of the current path."""
        if isinstance(message, Resp):
            # Since Resp is now a Msg, we can add it directly
            self.append(message)
            return message
        self.append(message)
        return message

    @abstractmethod
    def replace(
        self, message: t.Union[Msg, Resp], ind: int
    ) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def insert(self, message: Msg, ind: int) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def extend(
        self, dialog: t.Union['BaseDialog', t.Iterable[t.Union[Msg, Resp]]], 
        _inplace: bool=False
    ) -> 'BaseDialog':
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog: The dialog or list of messages to extend with
        """
        pass

    def render(self) -> str:
        """Render the dialog as a series of turns 
        <role>: <text>

        Returns:
            str: The dialog
        """
        if self._renderer is not None:
            return self._renderer(self)
        return '\n'.join(
            message.render()
            for message in self
        )

    def aslist(self) -> t.List['Msg']:
        """Retrieve the message list

        Returns:
            typing.List[Msg]: the messages in the dialog
        """
        return list(self)
    
    @abstractmethod
    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        pass
        
    @abstractmethod
    def clone(self) -> 'BaseDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        pass

    @abstractmethod
    def spawn(self) -> 'BaseDialog':
        """Create a new empty dialog of the same type as the current dialog.

        Returns:
            Dialog: A new empty dialog of the same type
        """
        pass


class ListDialog(BaseDialog):
    """A Dialog that uses a list data structure.
    """

    messages: t.List[Msg] = pydantic.Field(default_factory=list)

    def __iter__(self) -> t.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self.messages:
            yield message

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        return self.messages[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        if idx == len(self.messages):
            self.messages.append(message)
        else:
            self.messages[idx] = message
        return self

    def clone(self) -> 'ListDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        
        return ListDialog(
            messages=[*self.messages]
        )
    
    def spawn(self) -> 'ListDialog':
        """Create a new empty dialog of the same type as the current dialog.

        Returns:
            Dialog: A new empty dialog of the same type
        """
        return ListDialog(messages=[])

    def pop(self, index: int=-1, get_msg: bool=False) -> t.Union['ListDialog', Msg]:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        msg = self.messages.pop(index)
        if get_msg:
            return self.messages, msg
        return msg

    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        self.messages.remove(message)

    def extend(self, dialog: t.Iterable[t.Union[Msg, Resp]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog: The dialog or list of messages to extend with
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        validated = []
        for msg in dialog:
            if isinstance(msg, Resp):
                # Since Resp is now a Msg, we can add it directly
                validated.append(msg)
            elif isinstance(msg, Msg):
                validated.append(msg)
            else:
                raise ValueError(
                    "List dialog must only consist of messages."
                )

        self.messages.extend(validated)
        return self

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self.messages)
        
    def append(self, message: t.Union[Msg, Resp]) -> Self:
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        if isinstance(message, (Msg, Resp)):  # Resp is now a Msg
            self.messages.append(message)
        else:
            raise ValueError(
                "List dialog must only consist of messages."
            )
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        self.messages.insert(ind, message)
        return self

    def replace(self, idx: int, message: t.Union[Msg, Resp]) -> 'BaseDialog':
        """Replace a message in the dialog

        Args:
            idx (int): The index to replace at
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        if isinstance(message, (Msg, Resp)):  # Resp is now a Msg
            self.messages[idx] = message
        else:
            raise ValueError(
                "List dialog must only consist of messages."
            )
        return self


def exclude_messages(
    dialog: BaseDialog, 
    val: t.Union[t.Any, t.Set], 
    field='role'
) -> ListDialog:
    """Exclude messages from the dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to exclude based on
        field (str, optional): The field to exclude based on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, t.Set):
        val = {val}

    return ListDialog(
        messages=[msg for msg in dialog if getattr(msg, field) not in val]
    )

            
def include_messages(
    dialog: BaseDialog, 
    val: t.Union[t.Any, t.Set], 
    field='role'
) -> ListDialog:
    """Include messages in the resulting dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to include based on
        field (str, optional): The field to include based on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, t.Set):
        val = {val}

    return ListDialog(
        messages=[msg for msg in dialog if getattr(msg, field) in val]
    )


def to_dialog(prompt: t.Union[BaseDialog, Msg]) -> BaseDialog:
    """Convert a prompt to a dialog

    Args:
        prompt (typing.Union[Dialog, Msg]): The prompt to convert
    """
    if isinstance(prompt, Msg):
        prompt = ListDialog(messages=[prompt])

    return prompt


def exclude_role(
    messages: t.Iterable[Msg], 
    *role: str
) -> t.List[Msg]:
    """
    Filter messages by excluding specified roles.
    
    Args:
        messages (typing.Iterable[Msg]): An iterable of message objects
        *role (str): Variable number of role strings to exclude
    Returns:
        typing.List[Msg]: A list of messages excluding those with specified roles
    Example:
        >>> messages = [Msg(role="user", text="hi"), Msg(role="system", text="hello")]
        >>> exclude_role(messages, "system")
        [Msg(role="user", text="hi")]
    """
    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(
    messages: t.Iterable[Msg],
    *role: str
) -> t.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): Messages to filter

    Returns:
        typing.List[Msg]: Filtered messages
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]


class FieldRenderer(object):

    def __init__(self, field: str='text'):
        """Renderer to render a specific field in the message

        Args:
            field (str, optional): The field name. Defaults to 'text'.
        """
        self.field = field

    def __call__(self, msg: t.Union[Msg, BaseDialog]) -> str:
        """Render a message

        Args:
            msg (Msg): The message to render

        Returns:
            str: The result
        """
        if isinstance(msg, BaseDialog):
            messages = list(msg)
        else:
            messages = [msg]
        return '\n'.join(
            f'{m.role}: {getattr(m, self.field, "")}'
            for m in messages
        )


class TreeDialog(BaseDialog):
    """A tree-based dialog implementation using node ID mappings instead of DialogTurn objects.
    
    This implementation provides the same interface as TreeDialog but uses a more efficient
    mapping-based approach internally. Messages are stored in a tree structure where each
    node has a unique ID and relationships are maintained through parent/child mappings.
    
    Key Features:
        - Tree structure for conversation branching
        - Efficient node-based navigation
        - Same API as TreeDialog for compatibility
        - Support for insertion, replacement, and removal operations
        - Navigation through rise(), leaf_child(), leaf_sibling() methods
        - Maintains indices and counts for path tracking
        
    Internal Structure:
        - _messages: Maps node IDs to Msg objects
        - _parent: Maps child node IDs to parent node IDs  
        - _children: Maps parent node IDs to lists of child node IDs
        - _root: ID of the root node
        - _leaf: ID of the current leaf node being tracked
        - _indices: Current path indices from root to leaf
        - _counts: Number of children at each level in current path
    
    Usage:
        dialog = TreeDialog()
        dialog.append(Msg(role="user", text="Hello"))
        dialog.append(Msg(role="assistant", text="Hi there!"))
        
        # Navigate and branch
        dialog.rise(1)  # Go up one level
        dialog.append(Msg(role="assistant", text="Alternative response"))
        
        # Access messages
        for msg in dialog:
            print(msg.render())
    """
    _root: t.Optional[str] = PrivateAttr(default=None)
    _messages: t.Dict[str, Msg] = PrivateAttr(default_factory=dict)
    _leaf: t.Optional[str] = PrivateAttr(default=None)
    _parent: t.Dict[str, str] = PrivateAttr(default_factory=dict)
    _children: t.Dict[str, t.List[str]] = PrivateAttr(default_factory=dict)
    _indices: t.List[int] = PrivateAttr(default_factory=list)
    _counts: t.List[int] = PrivateAttr(default_factory=list)
    _next_id: int = PrivateAttr(default=0)
        
    def model_post_init(self, __context) -> None:
        """Initialize after model validation."""
        super().model_post_init(__context)
        self._update()
        
    def _generate_id(self) -> str:
        """Generate a unique node ID."""
        node_id = str(self._next_id)
        self._next_id += 1
        return node_id
        
    def _get_path_to_leaf(self) -> t.List[str]:
        """Get the path from root to current leaf as a list of node IDs."""
        if self._leaf is None:
            return []
        
        path = []
        current = self._leaf
        while current is not None:
            path.insert(0, current)
            current = self._parent.get(current)
        return path
        
    def _update_indices(self):
        """Update the indices array based on current leaf position."""
        if self._leaf is None:
            self._indices = []
            return
            
        path = self._get_path_to_leaf()
        indices = []
        
        for i in range(len(path)):
            if i == 0:
                indices.append(0)  # Root is always at index 0
            else:
                parent_id = path[i-1]
                child_id = path[i]
                children = self._children.get(parent_id, [])
                try:
                    idx = children.index(child_id)
                    indices.append(idx)
                except ValueError:
                    indices.append(0)
                    
        self._indices = indices
        
    def _update_counts(self):
        """Update the counts array based on current path."""
        if self._leaf is None:
            self._counts = []
            return
            
        path = self._get_path_to_leaf()
        counts = []
        
        for node_id in path:
            children = self._children.get(node_id, [])
            if len(children) == 0:
                counts.append(1)  # Leaf nodes count as 1
            else:
                counts.append(len(children))
                
        self._counts = counts
        
    def _update(self):
        """Update internal state after modifications."""
        self._update_indices()
        self._update_counts()
        
    @property
    def indices(self) -> t.List[int]:
        """Get a copy of the current path indices."""
        return [*self._indices]
        
    @property
    def counts(self) -> t.List[int]:
        """Get a copy of the current level counts."""
        return [*self._counts]
        
    @property
    def root(self) -> t.Optional[Msg]:
        """Get the root message."""
        if self._root is None:
            return None
        return self._messages[self._root]
        
    @property
    def leaf(self) -> t.Optional[Msg]:
        """Get the current leaf message."""
        if self._leaf is None:
            return None
        return self._messages[self._leaf]
        
    def rise(self, count: int):
        """Move the leaf pointer up by count levels."""
        if self._leaf is None:
            return
            
        current = self._leaf
        for _ in range(count):
            parent = self._parent.get(current)
            if parent is None:
                break
            current = parent
            
        self._leaf = current
        self._update()
        
    def leaf_child(self, idx: int):
        """Move to the specified child of the current leaf."""
        if self._leaf is None:
            return
            
        children = self._children.get(self._leaf, [])
        if 0 <= idx < len(children):
            self._leaf = children[idx]
            self._update()
            
    def leaf_sibling(self, idx: int):
        """Move to a sibling of the current leaf."""
        if self._leaf is None or self._leaf == self._root:
            return
            
        parent_id = self._parent.get(self._leaf)
        if parent_id is None:
            return
            
        siblings = self._children.get(parent_id, [])
        current_idx = siblings.index(self._leaf) if self._leaf in siblings else 0
        new_idx = current_idx + idx
        
        if 0 <= new_idx < len(siblings):
            self._leaf = siblings[new_idx]
            self._update()

    def __iter__(self) -> t.Iterator[Msg]:
        """Iterate over messages from root to current leaf."""
        if self._leaf is None:
            return
            
        path = self._get_path_to_leaf()
        for node_id in path:
            yield self._messages[node_id]

    def __getitem__(self, idx) -> Msg:
        """Get message at the specified index in the current path."""
        path = self._get_path_to_leaf()
        if 0 <= idx < len(path):
            return self._messages[path[idx]]
        raise IndexError("Index out of range")

    def __setitem__(self, idx, message) -> Self:
        """Set message at the specified index in the current path."""
        path = self._get_path_to_leaf()
        if 0 <= idx < len(path):
            self._messages[path[idx]] = message
        else:
            raise IndexError("Index out of range")
        return self

    def __len__(self) -> int:
        """Get the length of the current path from root to leaf."""
        return len(self._get_path_to_leaf())

    def pop(self, index: int = -1) -> Msg:
        """Remove and return a message at the specified index.
        
        Cannot remove the root node (index 0).
        """
        if index == 0:
            raise ValueError('Cannot remove root node in tree dialog.')
            
        path = self._get_path_to_leaf()
        if index == -1:
            index = len(path) - 1
            
        if not (0 <= index < len(path)):
            raise IndexError("Index out of range")
            
        target_id = path[index]
        target_msg = self._messages[target_id]
        
        # Remove the node and reconnect children to parent
        parent_id = self._parent.get(target_id)
        children_ids = self._children.get(target_id, [])
        
        # Update parent's children list
        if parent_id is not None:
            parent_children = self._children.get(parent_id, [])
            if target_id in parent_children:
                parent_children.remove(target_id)
                # Add the removed node's children to parent
                parent_children.extend(children_ids)
        
        # Update children's parent pointers
        for child_id in children_ids:
            self._parent[child_id] = parent_id
            
        # Clean up the removed node
        del self._messages[target_id]
        if target_id in self._parent:
            del self._parent[target_id]
        if target_id in self._children:
            del self._children[target_id]
            
        # Update leaf if it was the removed node
        if self._leaf == target_id:
            if parent_id is not None:
                self._leaf = parent_id
            elif children_ids:
                self._leaf = children_ids[0]
            else:
                self._leaf = None
                
        self._update()
        return target_msg

    def remove(self, message: Msg):
        """Remove a message from the dialog."""
        # Find the node ID for this message
        target_id = None
        for node_id, msg in self._messages.items():
            if msg == message:
                target_id = node_id
                break
                
        if target_id is None:
            raise ValueError(f'Message {message} does not exist so cannot remove.')
            
        if target_id == self._root:
            raise ValueError('Cannot remove root node in tree dialog.')
            
        # Remove using the node ID
        path = self._get_path_to_leaf()
        try:
            index = path.index(target_id)
            self.pop(index)
        except ValueError:
            # If not in current path, do direct removal
            parent_id = self._parent.get(target_id)
            children_ids = self._children.get(target_id, [])
            
            if parent_id is not None:
                parent_children = self._children.get(parent_id, [])
                if target_id in parent_children:
                    parent_children.remove(target_id)
                    parent_children.extend(children_ids)
            
            for child_id in children_ids:
                self._parent[child_id] = parent_id
                
            del self._messages[target_id]
            if target_id in self._parent:
                del self._parent[target_id]
            if target_id in self._children:
                del self._children[target_id]
                
            if self._leaf == target_id:
                if parent_id is not None:
                    self._leaf = parent_id
                elif children_ids:
                    self._leaf = children_ids[0]
                else:
                    self._leaf = None
                    
        self._update()

    def extend(
        self, 
        dialog: t.Union['BaseDialog', t.Iterable[t.Union[Msg, Resp]]], 
        _inplace: bool = False
    ) -> 'BaseDialog':
        """Extend the dialog with messages from another dialog or iterable."""
        if isinstance(dialog, BaseDialog):
            messages = list(dialog)
        else:
            messages = list(dialog)
            
        for msg in messages:
            self.append(msg)
            
        return self

    def append(self, message: t.Union[Msg, Resp]) -> Self:
        """Add a message to the end of the current path."""
        if isinstance(message, Resp):
            # Since Resp is now a Msg, we can use it directly
            pass
        elif not isinstance(message, Msg):
            raise ValueError("Message must be a Msg or Resp instance")
            
        node_id = self._generate_id()
        self._messages[node_id] = message
        
        if self._root is None:
            # First message becomes root
            self._root = node_id
            self._leaf = node_id
            self._children[node_id] = []
        else:
            # Add as child of current leaf
            self._parent[node_id] = self._leaf
            if self._leaf not in self._children:
                self._children[self._leaf] = []
            self._children[self._leaf].append(node_id)
            self._children[node_id] = []
            self._leaf = node_id
            
        self._update()
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Insert a message at the specified index in the current path."""
        if not isinstance(message, (Msg, Resp)):
            raise ValueError("Message must be a Msg or Resp instance")
            
        if self._leaf is None:
            if ind != 0:
                raise RuntimeError("Cannot insert at non-zero index in empty dialog")
            return self.append(message)
            
        path = self._get_path_to_leaf()
        
        if ind > len(path):
            raise IndexError(f"Index {ind} out of range for path of length {len(path)}")
        elif ind == len(path):
            # Append to end
            return self.append(message)
        elif ind == 0:
            # Insert before root - create new root
            new_id = self._generate_id()
            self._messages[new_id] = message
            
            # New node becomes root
            old_root = self._root
            self._root = new_id
            self._parent[old_root] = new_id
            self._children[new_id] = [old_root]
            
            self._update()
        else:
            # Insert in the middle - create branch
            target_id = path[ind]
            parent_id = self._parent.get(target_id)
            
            # Create new node
            new_id = self._generate_id()
            self._messages[new_id] = message
            
            # Insert between parent and target
            if parent_id is not None:
                parent_children = self._children.get(parent_id, [])
                target_index = parent_children.index(target_id)
                parent_children[target_index] = new_id
                
            self._parent[new_id] = parent_id
            self._parent[target_id] = new_id
            self._children[new_id] = [target_id]
            
            # Update root if necessary
            if target_id == self._root:
                self._root = new_id
                
            self._update()
            
        return self

    def replace(self, idx: int, message: t.Union[Msg, Resp]) -> 'BaseDialog':
        """Replace the message at the specified index."""
        if isinstance(message, (Msg, Resp)):
            pass  # Both are fine since Resp is now a Msg
        else:
            raise ValueError("Message must be a Msg or Resp instance")
            
        path = self._get_path_to_leaf()
        if not (0 <= idx < len(path)):
            raise ValueError(f"Index {idx} out of range for dialog of length {len(path)}")
            
        node_id = path[idx]
        self._messages[node_id] = message
        self._update()
        return self

    def clone(self) -> 'TreeDialog':
        """Create a deep copy of the dialog structure with shallow message copying."""
        clone = TreeDialog()
        
        if self._root is None:
            return clone
            
        # Copy all the mappings
        clone._messages = self._messages.copy()
        clone._parent = self._parent.copy()
        clone._children = {k: v.copy() for k, v in self._children.items()}
        clone._root = self._root
        clone._leaf = self._leaf
        clone._next_id = self._next_id
        
        clone._update()
        return clone
    
    def spawn(self):
        return TreeDialog()
