# 1st party
import typing as t
from typing import Literal

# 3rd party
import pydantic
import openai

# local
from ..core import (
    Msg, Resp, DeltaResp, BaseDialog
)
from ..core._tool import BaseTool
from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ._ai import LLMAdapter, LLM, extract_tools_from_messages, extract_format_override_from_messages, get_resp_output
from ._resp import ToOut


# Transformation helpers for OpenAI responses
def extract_commonly_useful_meta(output: t.Dict) -> t.Dict[str, t.Any]:
    """Extract commonly useful metadata fields for debugging, monitoring, and feature detection."""
    return {
        k: v for k, v in output.items() 
        if k in {
            "object", "created", "system_fingerprint", "service_tier"
        } and v is not None
    }


# Format conversion helper functions

def convert_tools_to_openai_format(tools: list[BaseTool]) -> list[dict]:
    """Convert Dachi tools to OpenAI tools format"""
    openai_tools = []
    for tool in tools:
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_model.model_json_schema()
            }
        }
        openai_tools.append(schema)
    return openai_tools


def _fix_schema_for_strict_mode(schema_dict: dict) -> None:
    """Recursively ensure all properties are in required array for OpenAI strict mode.

    OpenAI strict mode requires ALL properties to be in the 'required' array,
    even those with defaults. This function modifies the schema in-place.

    Args:
        schema_dict: JSON schema dictionary to fix
    """
    if not isinstance(schema_dict, dict):
        return

    # Fix this level
    if 'properties' in schema_dict:
        schema_dict['required'] = list(schema_dict['properties'].keys())

    # Recursively fix nested schemas in $defs
    if '$defs' in schema_dict:
        for def_schema in schema_dict['$defs'].values():
            _fix_schema_for_strict_mode(def_schema)


def build_openai_response_format(format_override, use_strict: bool = True) -> dict:
    """Convert format_override to OpenAI response_format (Chat API).

    Args:
        format_override: The format specification (Pydantic model, dict, bool, etc.)
        use_strict: Whether to use OpenAI's strict mode for Pydantic models.
                   When True (default), ensures schema compatibility with strict mode
                   by adding all properties to the 'required' array.

    Returns:
        dict: OpenAI response_format configuration
    """
    if format_override is None or format_override is False:
        return {}
    elif format_override is True or format_override == "json":
        return {"response_format": {"type": "json_object"}}
    elif format_override == "text":
        return {}  # Default text format
    elif isinstance(format_override, dict):
        # Dict is assumed to be a JSON schema
        # Check if it looks like a complete response_format or just a schema
        if 'type' in format_override and format_override.get('type') in ['json_object', 'json_schema']:
            # Already a response_format dict, pass through
            return {"response_format": format_override}
        else:
            # Treat as a JSON schema, wrap it for strict mode
            schema = format_override.copy()  # Don't modify original

            if use_strict:
                # Fix schema to meet OpenAI strict mode requirements
                _fix_schema_for_strict_mode(schema)

                # Need a name for the schema - use a generic one
                schema_name = schema.get('title', 'CustomSchema')

                return {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema
                        }
                    }
                }
            else:
                # Non-strict mode - use json_object
                return {"response_format": {"type": "json_object"}}
    elif isinstance(format_override, type) and issubclass(format_override, pydantic.BaseModel):
        # Convert Pydantic model class to JSON schema
        schema = format_override.model_json_schema()

        if use_strict:
            # Fix schema to meet OpenAI strict mode requirements (recursively)
            _fix_schema_for_strict_mode(schema)

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": format_override.__name__,
                    "strict": use_strict,
                    "schema": schema
                }
            }
        }
    else:
        raise ValueError(f"Unsupported format_override type: {type(format_override)}")


def build_openai_text_format(format_override) -> dict:
    """Convert format_override to OpenAI text.format (Responses API)"""
    if format_override is None or format_override is False:
        return {}
    elif format_override is True or format_override == "json":
        return {"text": {"format": {"type": "json_object"}}}
    elif format_override == "text":
        return {}  # Default text format
    elif isinstance(format_override, dict):
        return {"text": {"format": format_override}}
    elif isinstance(format_override, type) and issubclass(format_override, pydantic.BaseModel):
        # Convert Pydantic model class to JSON schema
        schema = format_override.model_json_schema()
        return {
            "text": {
                "format": {
                    "type": "json_schema",
                    "schema": schema
                }
            }
        }
    else:
        raise ValueError(f"Unsupported format_override type: {type(format_override)}")


def extract_openai_tool_calls(message: dict) -> list:
    """Extract tool calls from OpenAI response message"""
    return message.get("tool_calls", []) if message.get("tool_calls") else []


def accumulate_streaming_text(prev_text: str | None, delta_text: str | None) -> str:
    """Pure function for text accumulation without spawn()"""
    if prev_text is None:
        prev_text = ""
    if delta_text is None:
        delta_text = ""
    return prev_text + delta_text


class OpenAIChat(LLMAdapter):
    """
    Adapter for OpenAI Chat Completions API.
    
    Converts between Dachi's unified message format and OpenAI Chat Completions API format.
    
    Message Conversion:
    - Msg.role -> message.role (user/assistant/system/tool)
    - Msg.text -> message.content 
    - Msg.attachments -> message.content (for vision/multimodal)
    - Msg.tool_calls -> role="tool" messages with tool_call_id
    
    Streaming Pattern:
    1. Accumulates text without spawn() logic (pure accumulation)
    2. Returns both Resp and DeltaResp from streaming
    3. Uses helper functions for clean text accumulation
    
    Unified kwargs (converted):
    - temperature, max_tokens, top_p, frequency_penalty, presence_penalty
    - stream, stop, seed, user

    API-specific kwargs (passed through):
    - tools, tool_choice, response_format, logprobs, top_logprobs
    - parallel_tool_calls, service_tier, stream_options

    Dachi-specific kwargs:
    - use_strict (bool, default=True): Whether to use OpenAI's strict mode for
      structured outputs. When True, ensures all Pydantic model properties are
      marked as required in the schema, even if they have defaults.
    """

    def to_input(self, messages: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to Chat Completions format using universal helpers."""
        # Use universal helper functions to extract tools and format_override
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        # Convert single message to list for uniform processing
        if isinstance(messages, Msg):
            original_messages = [messages]
        else:
            original_messages = list(messages)
        
        # Convert messages to OpenAI format
        openai_messages = [self._convert_message(msg) for msg in original_messages]
        
        # Add tool messages from ToolUse objects (using original Msg objects)
        out_messages = []
        for i, msg in enumerate(openai_messages):
            out_messages.append(msg)
            # Access tool_calls from original Msg object, not converted dict
            for tool_out in original_messages[i].tool_calls:
                out_messages.append({
                    "role": "tool",
                    "content": str(tool_out.result),
                    "tool_call_id": tool_out.id
                })
        
        # Build final API input
        api_input = {
            "messages": out_messages,
            **kwargs
        }
        
        # Add tools if present
        if tools:
            api_input["tools"] = convert_tools_to_openai_format(tools)
            
        # Add response format if present
        if format_override:
            # Extract use_strict from kwargs if provided (default is True)
            use_strict = kwargs.pop('use_strict', True)
            api_input.update(build_openai_response_format(format_override, use_strict=use_strict))
        
        return api_input
    
    def from_result(
        self, 
        output: t.Dict | pydantic.BaseModel, 
        messages: Msg | BaseDialog
    ) -> Resp:
        """Convert Chat Completions response to Dachi Resp."""
        # Convert Pydantic model to dict if needed
        if isinstance(output, pydantic.BaseModel):
            output = output.model_dump()
        
        # Extract tools and format info from messages for potential validation/processing
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        choice = output.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        resp = Resp(
            role=message.get("role", "assistant"),
            text=message.get("content", ""),
            finish_reason=choice.get("finish_reason"),
            id=output.get("id"),
            model=output.get("model"),
            usage=output.get("usage") or {},
            tool_use=extract_openai_tool_calls(message),
            logprobs=choice.get("logprobs"),
            choices=[{
                "index": c.get("index", i),
                "finish_reason": c.get("finish_reason"),
                "logprobs": c.get("logprobs")
            } for i, c in enumerate(output.get("choices", []))]
        )
        
        # Store commonly useful metadata (debugging, monitoring, feature detection)
        resp.meta.update(extract_commonly_useful_meta(output))
        
        # Store raw response
        resp.raw = output
        return resp
    
    def from_streamed_result(
        self, 
        output: t.Dict | pydantic.BaseModel, 
        messages: Msg | BaseDialog, 
        prev_resp: Resp | None = None
    ) -> t.Tuple[Resp, DeltaResp]:
        """Handle Chat Completions streaming responses with pure accumulation."""
        # Convert Pydantic model to dict if needed
        if isinstance(output, pydantic.BaseModel):
            output = output.model_dump()
        
        # Extract tools and format info from messages for potential validation/processing
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        choice = output.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        
        # Accumulate text content using helper function
        delta_text = delta.get("content", "") or ""
        accumulated_text = accumulate_streaming_text(
            prev_resp.text if prev_resp else None, 
            delta_text
        )
        
        # Use previous role if delta doesn't specify one (common in streaming)
        role = delta.get("role")
        if role is None and prev_resp:
            role = prev_resp.role
        if role is None:
            role = "assistant"  # Default fallback
        
        # Create response with accumulated content (no spawn() logic)
        resp = Resp(
            role=role,
            text=accumulated_text,  # Full accumulated text
            finish_reason=choice.get("finish_reason"),
            id=output.get("id"),
            model=output.get("model"),
            usage=output.get("usage") or {}
        )
        
        # Create delta object for streaming
        delta_resp = DeltaResp(
            text=delta_text,  # Just the delta part
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason"),
            usage=output.get("usage")
        )
        
        # Store raw response
        resp.raw = output
        
        return resp, delta_resp
    
    def _convert_message(self, msg: Msg) -> t.Dict:
        """Convert single Msg to OpenAI message format."""
        openai_msg = {
            "role": msg.role,
            "content": msg.text or ""
        }
        
        # Handle attachments (vision)
        if msg.attachments:
            content = []
            if msg.text:
                content.append({"type": "text", "text": msg.text})
            
            for attachment in msg.attachments:
                if attachment.kind == "image":
                    image_url = attachment.data
                    if not image_url.startswith("data:"):
                        mime = attachment.mime or "image/png"
                        image_url = f"data:{mime};base64,{attachment.data}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            openai_msg["content"] = content
        
        return openai_msg
    


class OpenAIResp(LLMAdapter):
    """
    Adapter for OpenAI Responses API.
    
    Converts between Dachi's unified message format and OpenAI Responses API format.
    Specialized for reasoning models that provide thinking/reasoning content.
    
    Key Differences from Chat Completions:
    - Handles 'reasoning' field for model thinking process
    - Accumulates both text and thinking content separately during streaming
    - Uses pure accumulation without spawn() logic
    - Different parameter mapping (max_tokens -> max_output_tokens)
    
    Streaming Pattern:
    1. Accumulates text and thinking content using helper functions
    2. Returns both Resp and DeltaResp from streaming
    3. Uses text.format instead of response_format for structured output
    """
    
    def to_input(self, messages: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to Responses API format using universal helpers."""
        # Use universal helper functions to extract tools and format_override
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        # Handle single user message case - use simple string input
        if isinstance(messages, Msg) and messages.role == "user" and "instructions" not in kwargs:
            api_input = {
                "input": messages.text or "",
                **kwargs
            }
        else:
            # Handle multiple messages or complex cases - use input array format
            if isinstance(messages, Msg):
                original_messages = [messages]
            else:
                original_messages = list(messages)
            
            # Convert messages to OpenAI format
            openai_messages = [self._convert_message(msg) for msg in original_messages]
            
            # Add tool messages from ToolUse objects (using original Msg objects)
            out_messages = []
            for i, msg in enumerate(openai_messages):
                out_messages.append(msg)
                # Access tool_calls from original Msg object, not converted dict
                for tool_out in original_messages[i].tool_calls:
                    out_messages.append({
                        "role": "tool",
                        "content": str(tool_out.result),
                        "tool_call_id": tool_out.id
                    })
            
            api_input = {
                "input": out_messages,
                **kwargs
            }
        
        # Map parameter names for Responses API
        if 'max_tokens' in api_input:
            api_input['max_output_tokens'] = api_input.pop('max_tokens')
        
        # Add tools if present
        if tools:
            api_input["tools"] = convert_tools_to_openai_format(tools)
            
        # Add text format if present  
        if format_override:
            api_input.update(build_openai_text_format(format_override))
            
        return api_input
    
    def from_result(
        self, 
        output: t.Dict | pydantic.BaseModel, 
        messages: Msg | BaseDialog
    ) -> Resp:
        """Convert Responses API response to Dachi Resp."""
        if isinstance(output, pydantic.BaseModel):
            output = output.model_dump()
        
        # Extract tools and format info from messages for potential validation/processing
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        choice = output.get("choices", [{}])[0]
        message = choice.get("message", {})

        resp = Resp(
            role=message.get("role", "assistant"),
            text=message.get("content", ""),
            thinking=output.get("reasoning"),  # Can be string or dict
            finish_reason=choice.get("finish_reason"),
            id=output.get("id"),
            model=output.get("model"),
            usage=output.get("usage") or {},
            tool_use=extract_openai_tool_calls(message),
            logprobs=choice.get("logprobs"),
            choices=[{
                "index": c.get("index", i),
                "finish_reason": c.get("finish_reason"),
                "logprobs": c.get("logprobs")
            } for i, c in enumerate(output.get("choices", []))]
        )
        
        # Store commonly useful metadata (debugging, monitoring, feature detection)
        resp.meta.update(extract_commonly_useful_meta(output))
        
        # Store raw response
        resp.raw = output
        return resp
    
    def from_streamed_result(self, result: t.Dict, messages: Msg | BaseDialog, prev_resp: Resp | None = None) -> t.Tuple[Resp, DeltaResp]:
        """Convert streaming LLM response to Dachi Resp + DeltaResp"""
        # Convert Pydantic model to dict if needed
        if isinstance(result, pydantic.BaseModel):
            result = result.model_dump()
        
        # Extract tools and format info from messages for potential validation/processing
        tools = extract_tools_from_messages(messages)
        format_override = extract_format_override_from_messages(messages)
        
        choice = result.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if isinstance(messages, Msg):
            prev_id = messages.id
        else:
            prev_id = None

        # Accumulate text content
        delta_text = delta.get("content", "") or ""
        if prev_resp and prev_resp.text:
            accumulated_text = prev_resp.text + delta_text
        else:
            accumulated_text = delta_text

        # Accumulate thinking content - handle both string and dict formats
        delta_thinking = delta.get("reasoning")
        if prev_resp and prev_resp.thinking:
            if isinstance(prev_resp.thinking, str) and isinstance(delta_thinking, str):
                accumulated_thinking = prev_resp.thinking + (delta_thinking or "")
            elif isinstance(delta_thinking, str):
                # Previous was dict, current is string - just use current
                accumulated_thinking = delta_thinking
            else:
                # Keep previous if current is None/empty, otherwise use current
                accumulated_thinking = delta_thinking or prev_resp.thinking
        else:
            accumulated_thinking = delta_thinking

        # Use previous role if delta doesn't specify one (common in streaming)
        role = delta.get("role")
        if role is None and prev_resp:
            role = prev_resp.role
        if role is None:
            role = "assistant"  # Default fallback
        
        # Create accumulated response using new Resp inheritance model
        resp = Resp(
            role=role,
            text=accumulated_text,  # Full accumulated text
            id=result.get("id", None),
            prev_id=prev_id,
            thinking=accumulated_thinking,  # Full accumulated thinking
            model=result.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage=result.get("usage", {}),
            logprobs=choice.get("logprobs"),
            citations=choice.get("citations")
        )
        
        # Store raw response
        resp.raw = result
        resp.meta.update(extract_commonly_useful_meta(result))
        
        # Create delta object for streaming (just the incremental changes)
        delta_resp = DeltaResp(
            text=delta_text,  # Just the delta part
            thinking=delta.get("reasoning"),  # Just the delta part - can be string or dict
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason"),
            usage=result.get("usage")  # Can include nested structures
        )
        
        return resp, delta_resp
    
    def _convert_message(self, msg: Msg) -> t.Dict:
        """Convert single Msg to OpenAI message format."""
        openai_msg = {
            "role": msg.role,
            "content": msg.text or ""
        }
        
        # Handle attachments (same as Chat Completions)
        if msg.attachments:
            content = []
            if msg.text:
                content.append({"type": "text", "text": msg.text})
            
            for attachment in msg.attachments:
                if attachment.kind == "image":
                    image_url = attachment.data
                    if not image_url.startswith("data:"):
                        mime = attachment.mime or "image/png"
                        image_url = f"data:{mime};base64,{attachment.data}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            openai_msg["content"] = content
        
        return openai_msg
    


class OpenAILLM(LLM):
    """
    Complete LLM implementation for OpenAI that inherits from LLM base class.
    
    Combines an OpenAI adapter (Chat or Responses API) with an OpenAI client
    to provide a complete LLM interface.
    
    Fields:
        api: Choose between "chat" (Chat Completions) or "response" (Responses API)
        url: Optional custom base URL for the OpenAI API
        model: Model name (e.g., "gpt-4", "o1-preview")
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        **kwargs: Additional parameters passed to API calls
    
    Example:
        llm = OpenAILLM(api="chat", model="gpt-4", temperature=0.7)
        resp = llm.forward(Msg(role="user", text="Hello"))
    """

    api: Literal["chat", "response"] = "chat"
    url: str | None = None
    model: str = "gpt-5"
    api_key: str | None = None
    
    def __post_init__(self):
        """Initialize the OpenAI client and adapter after instance creation."""
        super().__post_init__()
        
        # Create OpenAI client
        client_kwargs = {}
        if self.url:
            client_kwargs["base_url"] = self.url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        
        # Create appropriate adapter
        if self.api == "chat":
            self._adapter = OpenAIChat()
        else:  # api == "response"
            self._adapter = OpenAIResp()
    
    def forward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
        """Execute LLM call synchronously.

        Tools should be attached to Prompt messages via prompt.tools.
        The adapter will extract them using extract_tools_from_messages().
        """
        # Add model to kwargs if not present
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Convert input to API format
        api_input = self._adapter.to_input(inp, **kwargs)
        
        # Make API call based on api type
        if self.api == "chat":
            result = self._client.chat.completions.create(**api_input)
        else:  # api == "response"
            result = self._client.responses.create(**api_input)
        
        # Convert result to Resp
        resp = self._adapter.from_result(result, inp)
        
        # Process output if specified
        if out is not None:
            resp.out = get_resp_output(resp, out)
        
        return resp
    
    async def aforward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
        """Execute LLM call asynchronously.

        Tools should be attached to Prompt messages via prompt.tools.
        The adapter will extract them using extract_tools_from_messages().
        """
        # Add model to kwargs if not present
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Convert input to API format
        api_input = self._adapter.to_input(inp, **kwargs)
        
        # Make API call based on api type
        if self.api == "chat":
            result = await self._async_client.chat.completions.create(**api_input)
        else:  # api == "response"
            result = await self._async_client.responses.create(**api_input)
        
        # Convert result to Resp
        resp = self._adapter.from_result(result, inp)
        
        # Process output if specified
        if out is not None:
            resp.out = get_resp_output(resp, out)
        
        return resp
    
    def stream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        """Execute LLM call with streaming.

        Tools should be attached to Prompt messages via prompt.tools.
        The adapter will extract them using extract_tools_from_messages().
        """
        # Add model to kwargs if not present
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Convert input to API format
        api_input = self._adapter.to_input(inp, **kwargs)
        api_input["stream"] = True
        
        # Make streaming API call based on api type
        if self.api == "chat":
            stream = self._client.chat.completions.create(**api_input)
        else:  # api == "response"
            stream = self._client.responses.create(**api_input)
        
        # Process streaming results
        prev_resp = None
        for chunk in stream:
            resp, delta_resp = self._adapter.from_streamed_result(chunk, inp, prev_resp)
            
            # Process output if specified
            if out is not None:
                resp.out = get_resp_output(resp, out)
            
            prev_resp = resp
            yield resp, delta_resp
    
    async def astream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        """Execute LLM call with async streaming.

        Tools should be attached to Prompt messages via prompt.tools.
        The adapter will extract them using extract_tools_from_messages().
        """
        # Add model to kwargs if not present
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Convert input to API format
        api_input = self._adapter.to_input(inp, **kwargs)
        api_input["stream"] = True
        
        # Make streaming API call based on api type
        if self.api == "chat":
            stream = await self._async_client.chat.completions.create(**api_input)
        else:  # api == "response"
            stream = await self._async_client.responses.create(**api_input)
        
        # Process streaming results
        prev_resp = None
        async for chunk in stream:
            resp, delta_resp = self._adapter.from_streamed_result(chunk, inp, prev_resp)
            
            # Process output if specified
            if out is not None:
                resp.out = get_resp_output(resp, out)
            
            prev_resp = resp
            yield resp, delta_resp
