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
from ..core._tool import BaseTool, ToolBuffer, ToolChunk, ToolUse
from ._ai import LangEngine, LLM, extract_tools_from_messages, extract_format_override_from_messages, get_resp_output
from ._resp import ToOut
import json

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

class OpenAIExtractorMixin:

    def extract_openai_tool_calls(self, message: dict) -> list[dict]:
        pass

    def extract_openai_delta_tool_calls(self, prev_resp: Resp, delta: dict) -> list[dict]:
        pass

    def set_core_elements(self, resp: Resp, message: t.Dict) -> t.Dict[str, t.Any]:
        pass

    def set_core_delta_elements(self, cur_resp: Resp, cur_delta: DeltaResp, prev_resp: Resp, delta_message: dict) -> t.Dict[str, t.Any]:
        pass




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


# ============================================================================
# Tool Call Processing Helper Functions
# ============================================================================

def _is_valid_json(s: str) -> bool:
    """Check if a string is valid JSON.

    Args:
        s: String to validate

    Returns:
        True if valid JSON, False otherwise
    """
    if not s or not isinstance(s, str):
        return False
    try:
        import json
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False


def _accumulate_text(prev: str | None, delta: str | None) -> str:
    """Accumulate text from previous and delta.

    Args:
        prev: Previous accumulated text
        delta: New delta text

    Returns:
        Accumulated text
    """
    if prev is None:
        prev = ""
    if delta is None:
        delta = ""
    return prev + delta


def _get_staging(resp: Resp | None, key: str) -> dict:
    """Get staging data from previous response.

    Args:
        resp: Previous response (may be None)
        key: Key to look up in _raw dict

    Returns:
        Staging dict (empty if not found)
    """
    if resp is None:
        return {}
    if not hasattr(resp, '_raw') or resp._raw is None:
        return {}
    return resp._raw.get(key, {})


def _store_staging(resp: Resp, key: str, staging: dict) -> None:
    """Store staging data in response _raw dict.

    Args:
        resp: Response object to store in
        key: Key to store under in _raw dict
        staging: Staging data to store
    """
    if not hasattr(resp, '_raw') or resp._raw is None:
        resp._raw = {}
    resp._raw[key] = staging


def _extract_text_delta_chat(chunk: dict) -> str | None:
    """Extract text delta from Chat Completions chunk.

    Args:
        chunk: Chat completion chunk dict

    Returns:
        Text delta or None
    """
    choice = chunk.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    content = delta.get("content")
    return content if content else None


def _extract_text_delta_resp(event: dict) -> str | None:
    """Extract text delta from Responses API event.

    Args:
        event: Responses API event dict

    Returns:
        Text delta or None
    """
    if event.get("type") == "content.delta":
        delta_data = event.get("delta", {})
        text = delta_data.get("text")
        return text if text else None
    return None


def _extract_thinking_delta_resp(event: dict) -> str | None:
    """Extract thinking delta from Responses API event.

    Args:
        event: Responses API event dict

    Returns:
        Thinking delta or None
    """
    if event.get("type") == "response.output_item.added":
        item = event.get("item", {})
        if item.get("type") == "message":
            content_list = item.get("content", [])
            for content_item in content_list:
                if content_item.get("type") == "reasoning":
                    return content_item.get("reasoning", "")
    return None


def _extract_tool_deltas_chat(chunk: dict) -> list[dict]:
    """Extract tool call deltas from Chat Completions chunk.

    Args:
        chunk: Chat completion chunk dict

    Returns:
        List of tool call delta dicts (may be empty)
    """
    choice = chunk.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    tool_calls = delta.get("tool_calls")

    if not tool_calls:
        return []

    result = []
    for tc in tool_calls:
        tool_delta = {}
        if "index" in tc:
            tool_delta["index"] = tc["index"]
        if "id" in tc:
            tool_delta["id"] = tc["id"]
        if "type" in tc:
            tool_delta["type"] = tc["type"]

        if "function" in tc:
            func = tc["function"]
            tool_delta["function"] = {}
            if "name" in func:
                tool_delta["function"]["name"] = func["name"]
            if "arguments" in func:
                tool_delta["function"]["arguments"] = func["arguments"]

        result.append(tool_delta)

    return result


def _extract_tool_deltas_resp(event: dict) -> dict | None:
    """Extract tool call delta from Responses API event.

    Args:
        event: Responses API event dict

    Returns:
        Tool call delta dict or None
    """
    event_type = event.get("type")

    if event_type == "response.function_call_arguments.delta":
        return {
            "call_id": event.get("call_id"),
            "arguments_delta": event.get("delta")
        }
    elif event_type == "response.function_call_arguments.done":
        return {
            "call_id": event.get("call_id"),
            "name": event.get("name"),
            "arguments": event.get("arguments")
        }

    return None


def _extract_tool_calls_chat(message: dict) -> list[dict]:
    """Extract complete tool calls from Chat Completions message.

    Args:
        message: Message dict from Chat Completions response

    Returns:
        List of canonical tool call dicts
    """
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return []

    result = []
    for tc in tool_calls:
        canonical = {
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "")
            },
            "_complete": True
        }
        result.append(canonical)

    return result


def _extract_tool_calls_resp(output: list[dict]) -> list[dict]:
    """Extract complete tool calls from Responses API output items.

    Args:
        output: List of output items from Responses API

    Returns:
        List of canonical tool call dicts
    """
    result = []
    for item in output:
        if item.get("type") == "function_call":
            canonical = {
                "id": item.get("call_id"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "")
                },
                "_complete": True
            }
            result.append(canonical)

    return result


def accumulate_streaming_text(prev_text: str | None, delta_text: str | None) -> str:
    """Pure function for text accumulation without spawn()"""
    if prev_text is None:
        prev_text = ""
    if delta_text is None:
        delta_text = ""
    return prev_text + delta_text


def convert_messages(msg: Msg | BaseDialog | list[Msg] | str) -> list[dict]:
    """Convert Dachi messages to OpenAI Responses API message format."""
    if isinstance(msg, Msg):
        original_messages = [msg]
    elif isinstance(msg, str):
        original_messages = [Msg(role="user", text=msg)]
    else:
        original_messages = list(msg)
    
    openai_messages = []
    for original_msg in original_messages:
        openai_msg = {
            "role": original_msg.role,
            "content": original_msg.text or ""
        }
        
        # Handle attachments (same as Chat Completions)
        if original_msg.attachments:
            content = []
            if original_msg.text:
                content.append({"type": "text", "text": original_msg.text})
            
            for attachment in original_msg.attachments:
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
        
        openai_messages.append(openai_msg)
    
    return openai_messages


def to_api_input(messages: list[dict], tools: list[BaseTool], format_override, **kwargs) -> t.Dict:
    """Convert to OpenAI Responses API input format."""
    api_input = {
        "messages": messages,
        **kwargs
    }
    
    # Add tools if present
    if tools:
        api_input["tools"] = convert_tools_to_openai_format(tools)
        
    # Add text format if present
    if format_override:
        api_input.update(build_openai_text_format(format_override))
    
    if kwargs.get('stream', False) is True:
        api_input['stream'] = True
    
    return api_input



class OpenAIChat(LangEngine, OpenAIExtractorMixin):
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

    def to_input(self, msg: Msg | BaseDialog | str | list[Msg], **kwargs) -> t.Dict:
        """Convert Dachi messages to Chat Completions format using universal helpers."""
        # Use universal helper functions to extract tools and format_override
        tools = extract_tools_from_messages(msg)
        format_override = extract_format_override_from_messages(msg)

        msg = convert_messages(msg)
        api_input = self.to_api_input(
            msg, tools, format_override, **kwargs
        )
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
        # tools = extract_tools_from_messages(messages)
        # format_override = extract_format_override_from_messages(messages)
        
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
        
        # # Extract tools and format info from messages for potential validation/processing
        # tools = extract_tools_from_messages(messages)
        # format_override = extract_format_override_from_messages(messages)
        
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
    
    def set_core_elements(self, resp: "Resp", message: t.Dict) -> t.Dict[str, t.Any]:
        payload = message

        resp.raw = payload
        resp.id = payload.get("id")
        resp.model = payload.get("model")
        resp.usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}

        # assistant message
        choices = payload.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        msg = choice0.get("message") or {}

        resp.role = msg.get("role", "assistant")
        resp.text = msg.get("content") or ""

        return {"assistant_message": msg, "choice": choice0}

    def set_tools(self, resp: "Resp", message: dict) -> dict:
        """
        Chat Completions (non-streamed):
        payload["choices"][0]["message"]["tool_calls"] -> resp.tool_use
        """

        # get available tools from adapter (best-effort)
        tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
        tool_map = {t.name: t for t in tools}

        choices = message.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        msg = choice0.get("message") or {}

        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            resp.tool_use = []
            return {"tool_calls": []}

        out: list["ToolUse"] = []
        raw_completed: list[dict] = []

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue

            name = fn.get("name")
            args_text = fn.get("arguments", "")
            if not isinstance(name, str) or not isinstance(args_text, str):
                continue

            tool_def = tool_map.get(name)
            if tool_def is None:
                # unknown tool: let subclass decide what to do
                continue

            try:
                args = json.loads(args_text) if args_text.strip() else {}
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}

            tool_id = tc.get("id") or f"tool_{len(out)}"
            tool_use = tool_def.to_tool_call(tool_id=tool_id, **args)
            tool_use.id = tc.get("id")  # preserve provider id if present

            out.append(tool_use)
            raw_completed.append(tc)

        resp.tool_use = out

    def set_tools_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # chat.completion.chunk payload
    ) -> dict:
        """
        Chat Completions streaming:
        chunk["choices"][0]["delta"]["tool_calls"][...]
        Partial tool call data is kept in cur_delta.accumulation until completion.

        Contract:
        - cur_delta.tool stays None unless a tool call fully completes in this chunk.
        - newly completed ToolUse objects get appended to cur_resp.tool_use
        - accumulation is cleared once completion happens (no lingering partials).
        """

        # Always reset per-chunk output
        cur_delta.tool = None

        # Start from prior accumulated tool_use, then add newly completed ones
        cur_resp.tool_use = list(prev_resp.tool_use)

        # Ensure accumulation dict exists
        if not isinstance(cur_delta.accumulation, dict):
            cur_delta.accumulation = {}

        # Get/create ToolBuffer in accumulation (must persist across chunks via caller)
        buf = cur_delta.accumulation.get("tool_buffer")
        if buf is None:
            tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
            buf = ToolBuffer(tools=tools)
            cur_delta.accumulation["tool_buffer"] = buf
            cur_delta.accumulation["tool_buffer_emitted"] = 0

        emitted_n = cur_delta.accumulation.get("tool_buffer_emitted", 0)
        if not isinstance(emitted_n, int):
            emitted_n = 0

        # Extract chunk delta tool calls
        choices = delta_message.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish_reason = choice0.get("finish_reason")

        delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
        tool_calls_delta = delta.get("tool_calls") or []
        if not isinstance(tool_calls_delta, list) or not tool_calls_delta:
            return {"completed": []}

        # Append tool chunks into buffer
        for tc in tool_calls_delta:
            if not isinstance(tc, dict):
                continue

            call_index = tc.get("index")
            if not isinstance(call_index, int):
                continue

            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue

            chunk = ToolChunk(
                id=tc.get("id") if isinstance(tc.get("id"), str) else None,
                turn_index=0,                  # chat doesn't expose a message index; stable constant is fine
                call_index=call_index,
                name=fn.get("name") if isinstance(fn.get("name"), str) else None,
                args_text_delta=fn.get("arguments") if isinstance(fn.get("arguments"), str) else None,
                done=(finish_reason == "tool_calls"),
            )
            buf.append(chunk)

        # Emit any newly completed calls (buffer stores them in buf._calls)
        new_calls = buf._calls[emitted_n:]
        if new_calls:
            cur_resp.tool_use.extend(new_calls)
            cur_delta.accumulation["tool_buffer_emitted"] = emitted_n + len(new_calls)

            # DeltaResp.tool must only be set when fully accumulated.
            # If multiple complete at once, store the last one's args as JSON text.
            last = new_calls[-1]
            try:
                cur_delta.tool = json.dumps(last.inputs.model_dump())
            except Exception:
                cur_delta.tool = "{}"

        # If tool calling finished this turn, clear accumulation buffer (your "reset" requirement)
        if finish_reason == "tool_calls":
            cur_delta.accumulation.pop("tool_buffer", None)
            cur_delta.accumulation.pop("tool_buffer_emitted", None)

    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # full chat.completion.chunk payload
    ) -> t.Dict[str, t.Any]:
        # --- delta: reset to "this chunk only"
        cur_delta.text = None
        cur_delta.usage = None
        cur_delta.finish_reason = None

        # --- extract
        choices = delta_message.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}

        delta = choice0.get("delta")
        delta = delta if isinstance(delta, dict) else {}

        text_inc = delta.get("content")          # incremental text fragment
        role_inc = delta.get("role")             # may appear only once (first chunk)
        finish_inc = choice0.get("finish_reason")  # usually only last chunk

        # usage may appear only if stream_options include it; treat as "latest total"
        usage_inc = delta_message.get("usage")
        usage_inc = usage_inc if isinstance(usage_inc, dict) else None

        # --- accumulate into cur_resp (prev + increment)
        cur_resp.id = prev_resp.id or delta_message.get("id")
        cur_resp.model = prev_resp.model or delta_message.get("model")

        cur_resp.role = (role_inc if isinstance(role_inc, str) and role_inc else prev_resp.role) or "assistant"

        prev_text = prev_resp.text if isinstance(prev_resp.text, str) else ""
        if isinstance(text_inc, str) and text_inc:
            cur_delta.text = text_inc
            cur_resp.text = prev_text + text_inc
        else:
            cur_resp.text = prev_text

        # usage: overwrite with latest if provided, otherwise keep previous
        cur_resp.usage = usage_inc if usage_inc is not None else (prev_resp.usage or {})
        if usage_inc is not None:
            cur_delta.usage = usage_inc

        # finish_reason exists for chat completions
        cur_resp.finish_reason = finish_inc if finish_inc is not None else prev_resp.finish_reason
        if finish_inc is not None:
            cur_delta.finish_reason = finish_inc

        # keep raw around (optional but often useful)
        cur_resp.raw = delta_message


class OpenAIResp(LangEngine, OpenAIExtractorMixin):
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

    def to_input(self, msg: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to Responses API format using universal helpers."""
        # Use universal helper functions to extract tools and format_override
        tools = extract_tools_from_messages(msg)
        format_override = extract_format_override_from_messages(msg)

        messages = self.convert_messages(msg)
        api_input = self.to_api_input(
            messages, tools, format_override, **kwargs
        )
        return api_input
    
    def _responses_text_from_message_item(msg_item: t.Dict[str, t.Any]) -> str:
        content = msg_item.get("content", [])
        if not isinstance(content, list):
            return ""
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )

    def set_core_elements(self, resp: "Resp", message: t.Dict) -> t.Dict[str, t.Any]:
        payload = message

        resp.raw = payload
        resp.id = payload.get("id")
        resp.model = payload.get("model")
        resp.usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}

        # first output message item
        output = payload.get("output") or []
        msg_item = next(
            (it for it in output if isinstance(it, dict) and it.get("type") == "message"),
            {}
        )

        resp.role = msg_item.get("role", "assistant")
        resp.text = _responses_text_from_message_item(msg_item)

    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ) -> t.Dict[str, t.Any]:
        cur_delta.text = None
        cur_delta.usage = None
        cur_delta.finish_reason = None  # no clean analogue here; leave to subclass if desired

        ev_type = delta_message.get("type")

        # Some events contain a nested "response" object (created/completed/error-ish)
        resp_obj = delta_message.get("response")
        resp_obj = resp_obj if isinstance(resp_obj, dict) else {}

        # Accumulate identity/model as they become available
        cur_resp.id = prev_resp.id or delta_message.get("response_id") or resp_obj.get("id")
        cur_resp.model = prev_resp.model or resp_obj.get("model")
        cur_resp.role = prev_resp.role or "assistant"

        # Text is typically delivered via response.output_text.delta events
        prev_text = prev_resp.text if isinstance(prev_resp.text, str) else ""
        if ev_type == "response.output_text.delta":
            text_inc = delta_message.get("delta")
            if isinstance(text_inc, str) and text_inc:
                cur_delta.text = text_inc
                cur_resp.text = prev_text + text_inc
            else:
                cur_resp.text = prev_text
        else:
            cur_resp.text = prev_text

        # Usage often arrives on/near completion in resp_obj["usage"]; treat as "latest total"
        usage_inc = resp_obj.get("usage")
        usage_inc = usage_inc if isinstance(usage_inc, dict) else None

        cur_resp.usage = usage_inc if usage_inc is not None else (prev_resp.usage or {})
        if usage_inc is not None:
            cur_delta.usage = usage_inc

        cur_resp.raw = delta_message
            


        # # Handle single user message case - use simple string input
        # if isinstance(msg, Msg) and msg.role == "user" and "instructions" not in kwargs:
        #     api_input = {
        #         "input": msg.text or "",
        #         **kwargs
        #     }
        # else:
        #     # Handle multiple messages or complex cases - use input array format
        #     if isinstance(msg, Msg):
        #         original_messages = [msg]
        #     else:
        #         original_messages = list(msg)
            
        #     # Convert messages to OpenAI format
        #     openai_messages = [self._convert_message(msg) for msg in original_messages]
            
        #     # Add tool messages from ToolUse objects (using original Msg objects)
        #     out_messages = []
        #     for i, msg in enumerate(openai_messages):
        #         out_messages.append(msg)
        #         # Access tool_calls from original Msg object, not converted dict
        #         for tool_out in original_messages[i].tool_calls:
        #             out_messages.append({
        #                 "role": "tool",
        #                 "content": str(tool_out.result),
        #                 "tool_call_id": tool_out.id
        #             })
            
        #     api_input = {
        #         "input": out_messages,
        #         **kwargs
        #     }
        
        # # Map parameter names for Responses API
        # if 'max_tokens' in api_input:
        #     api_input['max_output_tokens'] = api_input.pop('max_tokens')
        
        # # Add tools if present
        # if tools:
        #     api_input["tools"] = convert_tools_to_openai_format(tools)
            
        # # Add text format if present  
        # if format_override:
        #     api_input.update(build_openai_text_format(format_override))
        
        # if kwargs.get('stream', False) is True:
        #     api_input['stream'] = True
            
        # return api_input
    
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
    
        text = 
        
        citations = self.extract_citations(
            result, delta=prev_resp
        )
        thinking = extract_delta_thinking(result, messages, prev_resp)
        tool_use = extract_delta_tool_use(result, messages, prev_resp)

    def set_core_elements(self, resp: Resp, message: t.Dict) -> t.Dict[str, t.Any]:
        # Set all of the core elements here
        
        resp.text = message.get("content", "")
        resp.role = message.get("role", "assistant")

    def set_tools(self, resp: "Resp", message: dict) -> dict:
        """
        Responses API (non-streamed):
        payload["output"] contains items; tool calls are typically items with type == "function_call".
        Completed tool calls -> resp.tool_use (ToolUse objects).
        """

        tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
        tool_map = {t.name: t for t in tools}

        output = message.get("output") or []
        if not isinstance(output, list) or not output:
            resp.tool_use = []
            return {"tool_calls": []}

        out: list["ToolUse"] = []
        raw_completed: list[dict] = []

        for item in output:
            if not isinstance(item, dict):
                continue

            itype = item.get("type")
            if itype not in ("function_call", "tool_call"):
                continue

            # Responses commonly uses: {"type":"function_call","call_id":...,"name":...,"arguments":...}
            name = item.get("name") or item.get("tool_name")
            args_text = item.get("arguments", "")
            call_id = item.get("call_id") or item.get("id")

            if not isinstance(name, str) or not isinstance(args_text, str):
                continue

            tool_def = tool_map.get(name)
            if tool_def is None:
                continue

            try:
                args = json.loads(args_text) if args_text.strip() else {}
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}

            tool_id = (call_id if isinstance(call_id, str) and call_id else f"tool_{len(out)}")
            tool_use = tool_def.to_tool_call(tool_id=tool_id, **args)

            # Preserve provider ids where possible
            if isinstance(item.get("id"), str):
                tool_use.id = item["id"]

            out.append(tool_use)
            raw_completed.append(item)

        resp.tool_use = out

    def set_tools_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ) -> dict:
        """
        Responses API (streaming events):
        - Accumulate partial tool calls in cur_delta.accumulation via ToolBuffer
        - Only set cur_delta.tool when a tool call has fully completed
        - When a call completes, ToolBuffer clears that call’s accumulator automatically
        - We also clear the item_id routing entry on completion
        """

        # DeltaResp.tool must be None unless we completed a full tool call this chunk
        cur_delta.tool = None

        # Start tool_use accumulation from prev_resp, then append any newly completed calls
        cur_resp.tool_use = list(prev_resp.tool_use)

        # Ensure accumulation exists
        if not isinstance(cur_delta.accumulation, dict):
            cur_delta.accumulation = {}

        # ToolBuffer + emitted counter persist in cur_delta.accumulation (your caller must carry it forward)
        buf = cur_delta.accumulation.get("tool_buffer")
        if buf is None:
            tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
            buf = ToolBuffer(tools=tools)
            cur_delta.accumulation["tool_buffer"] = buf
            cur_delta.accumulation["tool_emitted"] = 0
            cur_delta.accumulation["tool_item_meta"] = {}  # item_id -> {"call_id":..., "name":...}

        emitted = cur_delta.accumulation.get("tool_emitted", 0)
        if not isinstance(emitted, int):
            emitted = 0
            cur_delta.accumulation["tool_emitted"] = 0

        item_meta = cur_delta.accumulation.get("tool_item_meta")
        if not isinstance(item_meta, dict):
            item_meta = {}
            cur_delta.accumulation["tool_item_meta"] = item_meta

        ev_type = delta_message.get("type")

        # ---- 1) Discover tool call identity (name/call_id) when item is added/done
        if ev_type in ("response.output_item.added", "response.output_item.done"):
            item = delta_message.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = item.get("id") if isinstance(item.get("id"), str) else None
                call_id = item.get("call_id") if isinstance(item.get("call_id"), str) else None
                name = item.get("name") if isinstance(item.get("name"), str) else None
                args_text = item.get("arguments") if isinstance(item.get("arguments"), str) else None

                if item_id and name:
                    # store routing so later arguments.delta knows the name/call_id
                    item_meta[item_id] = {"call_id": (call_id or item_id), "name": name}

                    # feed any args already present
                    buf.append(
                        ToolChunk(
                            id=(call_id or item_id),
                            name=name,
                            args_text_delta=args_text,
                            done=(ev_type == "response.output_item.done"),
                            metadata={"event": ev_type, "item_id": item_id},
                        )
                    )

                    # if done, clear routing entry now
                    if ev_type == "response.output_item.done":
                        item_meta.pop(item_id, None)

        # ---- 2) Stream arguments fragments
        elif ev_type == "response.function_call_arguments.delta":
            item_id = delta_message.get("item_id") if isinstance(delta_message.get("item_id"), str) else None
            text = delta_message.get("delta") if isinstance(delta_message.get("delta"), str) else None

            if item_id and text:
                meta = item_meta.get(item_id) or {}
                call_id = meta.get("call_id") if isinstance(meta.get("call_id"), str) else item_id
                name = meta.get("name") if isinstance(meta.get("name"), str) else None

                if name:
                    buf.append(
                        ToolChunk(
                            id=call_id,
                            name=name,
                            args_text_delta=text,
                            done=False,
                            metadata={"event": ev_type, "item_id": item_id},
                        )
                    )

        # ---- 3) Mark arguments done
        elif ev_type == "response.function_call_arguments.done":
            item_id = delta_message.get("item_id") if isinstance(delta_message.get("item_id"), str) else None
            final_text = delta_message.get("arguments") if isinstance(delta_message.get("arguments"), str) else None

            if item_id:
                meta = item_meta.get(item_id) or {}
                call_id = meta.get("call_id") if isinstance(meta.get("call_id"), str) else item_id
                name = meta.get("name") if isinstance(meta.get("name"), str) else None

                if name:
                    buf.append(
                        ToolChunk(
                            id=call_id,
                            name=name,
                            args_text_delta=final_text,
                            done=True,
                            metadata={"event": ev_type, "item_id": item_id},
                        )
                    )

                # routing no longer needed once done signal arrives
                item_meta.pop(item_id, None)

        # ---- Emit any newly completed ToolUse objects from ToolBuffer
        new_calls = buf._calls[emitted:]
        if new_calls:
            cur_resp.tool_use.extend(new_calls)
            cur_delta.accumulation["tool_emitted"] = emitted + len(new_calls)

            # DeltaResp.tool must only be set when fully accumulated.
            # ToolBuffer doesn’t keep exact original JSON text, so we emit canonical JSON from inputs.
            last = new_calls[-1]
            try:
                cur_delta.tool = json.dumps(last.inputs.model_dump())
            except Exception:
                cur_delta.tool = "{}"

        return {"completed": new_calls}


        # Extract tools and format info from messages for potential validation/processing
        # tools = extract_tools_from_messages(messages)
        # format_override = extract_format_override_from_messages(messages)
        
        # choice = result.get("choices", [{}])[0]
        # delta = choice.get("delta", {})

        # if isinstance(messages, Msg):
        #     prev_id = messages.id
        # else:
        #     prev_id = None

        # # Accumulate text content
        # delta_text = delta.get("content", "") or ""
        # if prev_resp and prev_resp.text:
        #     accumulated_text = prev_resp.text + delta_text
        # else:
        #     accumulated_text = delta_text

        # # Accumulate thinking content - handle both string and dict formats
        # delta_thinking = delta.get("reasoning")
        # if prev_resp and prev_resp.thinking:
        #     if isinstance(prev_resp.thinking, str) and isinstance(delta_thinking, str):
        #         accumulated_thinking = prev_resp.thinking + (delta_thinking or "")
        #     elif isinstance(delta_thinking, str):
        #         # Previous was dict, current is string - just use current
        #         accumulated_thinking = delta_thinking
        #     else:
        #         # Keep previous if current is None/empty, otherwise use current
        #         accumulated_thinking = delta_thinking or prev_resp.thinking
        # else:
        #     accumulated_thinking = delta_thinking

        # # Use previous role if delta doesn't specify one (common in streaming)
        # role = delta.get("role")
        # if role is None and prev_resp:
        #     role = prev_resp.role
        # if role is None:
        #     role = "assistant"  # Default fallback
        
        # # Create accumulated response using new Resp inheritance model
        # resp = Resp(
        #     role=role,
        #     text=accumulated_text,  # Full accumulated text
        #     id=result.get("id", None),
        #     prev_id=prev_id,
        #     thinking=accumulated_thinking,  # Full accumulated thinking
        #     model=result.get("model"),
        #     finish_reason=choice.get("finish_reason"),
        #     usage=result.get("usage", {}),
        #     logprobs=choice.get("logprobs"),
        #     citations=choice.get("citations")
        # )
        
        # # Store raw response
        # resp.raw = result
        # resp.meta.update(extract_commonly_useful_meta(result))
        
        # # Create delta object for streaming (just the incremental changes)
        # delta_resp = DeltaResp(
        #     text=delta_text,  # Just the delta part
        #     thinking=delta.get("reasoning"),  # Just the delta part - can be string or dict
        #     tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
        #     finish_reason=choice.get("finish_reason"),
        #     usage=result.get("usage")  # Can include nested structures
        # )
        
        # return resp, delta_resp
    
    # def _convert_message(self, msg: Msg) -> t.Dict:
    #     """Convert single Msg to OpenAI message format."""
    #     openai_msg = {
    #         "role": msg.role,
    #         "content": msg.text or ""
    #     }
        
    #     # Handle attachments (same as Chat Completions)
    #     if msg.attachments:
    #         content = []
    #         if msg.text:
    #             content.append({"type": "text", "text": msg.text})
            
    #         for attachment in msg.attachments:
    #             if attachment.kind == "image":
    #                 image_url = attachment.data
    #                 if not image_url.startswith("data:"):
    #                     mime = attachment.mime or "image/png"
    #                     image_url = f"data:{mime};base64,{attachment.data}"
                    
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {"url": image_url}
    #                 })
            
    #         openai_msg["content"] = content
        
    #     return openai_msg
    
import typing as t

def _responses_text_from_message_item(msg_item: t.Dict[str, t.Any]) -> str:
content = msg_item.get("content", [])
if not isinstance(content, list):
    return ""
return "".join(
    part.get("text", "")
    for part in content
    if isinstance(part, dict) and isinstance(part.get("text"), str)
)


# class OpenAILLM(LLM):
#     """
#     Complete LLM implementation for OpenAI that inherits from LLM base class.
    
#     Combines an OpenAI adapter (Chat or Responses API) with an OpenAI client
#     to provide a complete LLM interface.
    
#     Fields:
#         api: Choose between "chat" (Chat Completions) or "response" (Responses API)
#         url: Optional custom base URL for the OpenAI API
#         model: Model name (e.g., "gpt-4", "o1-preview")
#         api_key: Optional API key (defaults to OPENAI_API_KEY env var)
#         **kwargs: Additional parameters passed to API calls
    
#     Example:
#         llm = OpenAILLM(api="chat", model="gpt-4", temperature=0.7)
#         resp = llm.forward(Msg(role="user", text="Hello"))
#     """

#     api: Literal["chat", "response"] = "chat"
#     url: str | None = None
#     model: str = "gpt-5"
#     api_key: str | None = None
#     _client: openai.OpenAI = pydantic.PrivateAttr()
#     _async_client: openai.AsyncOpenAI = pydantic.PrivateAttr()
#     _adapter: LLMAdapter = pydantic.PrivateAttr()
    
#     def model_post_init(self, __context):
#         """Initialize the OpenAI client and adapter after instance creation."""
#         super().model_post_init(__context)
        
#         # Create OpenAI client
#         client_kwargs = {}
#         if self.url:
#             client_kwargs["base_url"] = self.url
#         if self.api_key:
#             client_kwargs["api_key"] = self.api_key
        
#         self._client = openai.OpenAI(**client_kwargs)
#         self._async_client = openai.AsyncOpenAI(**client_kwargs)
        
#         # Create appropriate adapter
#         if self.api == "chat":
#             self._adapter = OpenAIChat()
#         else:  # api == "response"
#             self._adapter = OpenAIResp()
    
#     def forward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
#         """Execute LLM call synchronously.

#         Tools should be attached to Prompt messages via prompt.tools.
#         The adapter will extract them using extract_tools_from_messages().
#         """
#         # Add model to kwargs if not present
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         # Convert input to API format
#         api_input = self._adapter.to_input(inp, **kwargs)
        
#         # Make API call based on api type
#         if self.api == "chat":
#             result = self._client.chat.completions.create(**api_input)
#         else:  # api == "response"
#             result = self._client.responses.create(**api_input)
        
#         # Convert result to Resp
#         resp = self._adapter.from_result(result, inp)
        
#         # Process output if specified
#         if out is not None:
#             resp.out = get_resp_output(resp, out)
        
#         return resp
    
#     async def aforward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
#         """Execute LLM call asynchronously.

#         Tools should be attached to Prompt messages via prompt.tools.
#         The adapter will extract them using extract_tools_from_messages().
#         """
#         # Add model to kwargs if not present
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         # Convert input to API format
#         api_input = self._adapter.to_input(inp, **kwargs)
        
#         # Make API call based on api type
#         if self.api == "chat":
#             result = await self._async_client.chat.completions.create(**api_input)
#         else:  # api == "response"
#             result = await self._async_client.responses.create(**api_input)
        
#         # Convert result to Resp
#         resp = self._adapter.from_result(result, inp)
        
#         # Process output if specified
#         if out is not None:
#             resp.out = get_resp_output(resp, out)
        
#         return resp
    
#     def stream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
#         """Execute LLM call with streaming.

#         Tools should be attached to Prompt messages via prompt.tools.
#         The adapter will extract them using extract_tools_from_messages().
#         """
#         # Add model to kwargs if not present
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         # Convert input to API format
#         api_input = self._adapter.to_input(inp, **kwargs)
#         api_input["stream"] = True
        
#         # Make streaming API call based on api type
#         if self.api == "chat":
#             stream = self._client.chat.completions.create(**api_input)
#         else:  # api == "response"
#             stream = self._client.responses.create(**api_input)
        
#         # Process streaming results
#         prev_resp = None
#         for chunk in stream:
#             resp, delta_resp = self._adapter.from_streamed_result(chunk, inp, prev_resp)
            
#             # Process output if specified
#             if out is not None:
#                 resp.out = get_resp_output(resp, out)
            
#             prev_resp = resp
#             yield resp, delta_resp
    
#     async def astream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
#         """Execute LLM call with async streaming.

#         Tools should be attached to Prompt messages via prompt.tools.
#         The adapter will extract them using extract_tools_from_messages().
#         """
#         # Add model to kwargs if not present
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         # Convert input to API format
#         api_input = self._adapter.to_input(inp, **kwargs)
#         api_input["stream"] = True
        
#         # Make streaming API call based on api type
#         if self.api == "chat":
#             stream = await self._async_client.chat.completions.create(**api_input)
#         else:  # api == "response"
#             stream = await self._async_client.responses.create(**api_input)
        
#         # Process streaming results
#         prev_resp = None
#         async for chunk in stream:
#             resp, delta_resp = self._adapter.from_streamed_result(chunk, inp, prev_resp)
            
#             # Process output if specified
#             if out is not None:
#                 resp.out = get_resp_output(resp, out)
            
#             prev_resp = resp
#             yield resp, delta_resp



# Below is a practical, **streaming-safe** way to reconstruct tool calls as they arrive in deltas—for **both** the **Responses API** and the **Chat Completions API**—plus what you must extract from each.

# ## What you must extract (per API)

# ### Responses API (streaming)

# A tool call is represented as an **output item of type `function_call`** in `response.output` (non-streamed), e.g. it includes `id`, `call_id`, `name`, `arguments`, `status`. ([GitHub][1])

# In streaming, you typically assemble it from:

# 1. **Creation / identity** (usually when the item appears)

# * Event: `response.output_item.added`
# * Extract from `event["item"]` (when `item["type"] == "function_call"`):

#   * `item["id"]` (the output-item id; often looks like `fc_...`)
#   * `item["call_id"]` (**the id you must return results against**)
#   * `item["name"]`
#   * `item.get("arguments")` (may be empty or already present)

# 2. **Arguments streaming**

# * Event: `response.function_call_arguments.delta`
# * Extract:

#   * `event["item_id"]` (which function_call item to update)
#   * `event["delta"]` (string fragment to append) ([GitHub][1])

# 3. **Arguments finalized**

# * Event: `response.function_call_arguments.done`
# * Extract:

#   * `event["item_id"]`
#   * `event["arguments"]` (final full JSON string)

# So in practice: **use `output_item.added` to learn `name/call_id`**, and **use `function_call_arguments.*` to build `arguments`**.

# ---

# ### Chat Completions API (streaming)

# A tool call is surfaced inside the assistant message delta as `choices[].delta.tool_calls[]` (and finalized in the non-streamed response as `choices[].message.tool_calls[]`). Streaming chunks are `chat.completion.chunk`. ([GitHub][1])

# From each streaming chunk, extract from (per choice):

# * `chunk["choices"][i]["delta"]["tool_calls"]` (list)

#   * each entry’s stable position: `tool_call_delta["index"]`
#   * `tool_call_delta.get("id")`
#   * `tool_call_delta.get("type")` (typically `"function"`)
#   * `tool_call_delta.get("function", {}).get("name")`
#   * `tool_call_delta.get("function", {}).get("arguments")` (string fragment; append)

# ---

# ## Common reconstruction strategy

# Regardless of API, you:

# 1. Identify **which tool call** the delta belongs to (Responses: `item_id`; Chat: `index`).
# 2. Ensure a corresponding tool-call object exists in `cur_message["tool_calls"]`.
# 3. Merge fields as they arrive:

#    * set `id`/`call_id`/`name` when first seen
#    * append `arguments` fragments
# 4. Consider a tool call “built” when:

#    * you have a `name`, and
#    * `arguments` is a complete JSON string (or you received a `*.done` event)

# ---

# ## Code: one dispatcher + two builders (Responses + Chat)

# ```python
# from __future__ import annotations

# from dataclasses import dataclass
# import json
# from typing import Any, Dict, List, Optional, Tuple


# ToolCall = Dict[str, Any]
# Message = Dict[str, Any]


# def _ensure_list_size(lst: List[Any], size: int, fill_factory):
#     while len(lst) < size:
#         lst.append(fill_factory())


# def _safe_json_loads(s: str) -> bool:
#     """Return True iff s is valid JSON (object/array/etc)."""
#     try:
#         json.loads(s)
#         return True
#     except Exception:
#         return False


# def _canonical_tool_call(*, call_id: Optional[str], name: Optional[str], arguments: str) -> ToolCall:
#     """
#     Canonicalize to the Chat Completions-style shape:
#       {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
#     For Responses API, we map response.item.call_id -> "id".
#     """
#     return {
#         "id": call_id,
#         "type": "function",
#         "function": {
#             "name": name,
#             "arguments": arguments,
#         },
#         # Internal bookkeeping (safe to remove if you prefer):
#         "_complete": False,
#     }


# def build_tool_calls_from_chat_completions_delta(
#     cur_message: Message,
#     delta: Dict[str, Any],
# ) -> Tuple[List[ToolCall], Optional[ToolCall]]:
#     """
#     `delta` here should be the per-choice delta dict:
#       chunk["choices"][i]["delta"]
#     """
#     tool_calls_delta = delta.get("tool_calls") or []
#     if not tool_calls_delta:
#         # Also handle deprecated "function_call" if present, by converting it:
#         # (many clients can ignore this in 2025; kept for robustness)
#         fc = delta.get("function_call")
#         if not fc:
#             return cur_message.get("tool_calls", []) or [], None

#         # Convert deprecated function_call -> tool_calls[0] shape.
#         tool_calls_delta = [{
#             "index": 0,
#             "id": None,
#             "type": "function",
#             "function": {
#                 "name": fc.get("name"),
#                 "arguments": fc.get("arguments", ""),
#             }
#         }]

#     tool_calls: List[ToolCall] = cur_message.setdefault("tool_calls", [])
#     # Track “was incomplete” by index to detect “newly built” when JSON becomes valid.
#     index_state: Dict[int, bool] = cur_message.setdefault("_tool_call_incomplete_by_index", {})

#     new_built: Optional[ToolCall] = None

#     for tc_delta in tool_calls_delta:
#         idx = tc_delta.get("index")
#         if idx is None:
#             # If no index, treat as append-only (rare); put at end.
#             idx = len(tool_calls)

#         _ensure_list_size(
#             tool_calls,
#             idx + 1,
#             lambda: _canonical_tool_call(call_id=None, name=None, arguments=""),
#         )

#         tc = tool_calls[idx]

#         # Merge id/type
#         if tc_delta.get("id") is not None:
#             tc["id"] = tc_delta["id"]
#         if tc_delta.get("type") is not None:
#             tc["type"] = tc_delta["type"]

#         # Merge function payload
#         fn_delta = tc_delta.get("function") or {}
#         fn = tc.setdefault("function", {"name": None, "arguments": ""})

#         if fn_delta.get("name") is not None:
#             fn["name"] = fn_delta["name"]

#         if fn_delta.get("arguments") is not None:
#             fn["arguments"] = (fn.get("arguments") or "") + fn_delta["arguments"]

#         # Decide if “built”
#         name = fn.get("name")
#         args = fn.get("arguments") or ""
#         now_complete = bool(name) and _safe_json_loads(args)

#         # Detect transition incomplete -> complete
#         was_incomplete = index_state.get(idx, True)
#         index_state[idx] = not now_complete

#         tc["_complete"] = now_complete
#         if now_complete and was_incomplete and new_built is None:
#             new_built = tc

#     return tool_calls, new_built


# def build_tool_calls_from_responses_stream_event(
#     cur_message: Message,
#     event: Dict[str, Any],
# ) -> Tuple[List[ToolCall], Optional[ToolCall]]:
#     """
#     `event` here is a single Responses streaming event dict, e.g.:
#       {"type": "response.output_item.added", ...}
#       {"type": "response.function_call_arguments.delta", ...}
#       {"type": "response.function_call_arguments.done", ...}
#     """
#     tool_calls: List[ToolCall] = cur_message.setdefault("tool_calls", [])
#     by_item_id: Dict[str, int] = cur_message.setdefault("_tool_call_index_by_item_id", {})

#     def ensure_item_slot(item_id: str) -> ToolCall:
#         if item_id in by_item_id:
#             return tool_calls[by_item_id[item_id]]
#         tool_calls.append(_canonical_tool_call(call_id=None, name=None, arguments=""))
#         by_item_id[item_id] = len(tool_calls) - 1
#         return tool_calls[-1]

#     t = event.get("type")
#     new_built: Optional[ToolCall] = None

#     # 1) Tool call item appears
#     if t == "response.output_item.added":
#         item = event.get("item") or {}
#         if item.get("type") != "function_call":
#             return tool_calls, None

#         item_id = item.get("id")
#         if not item_id:
#             return tool_calls, None

#         tc = ensure_item_slot(item_id)

#         # Responses function_call item provides these directly (non-streamed example shows call_id/name/arguments). :contentReference[oaicite:3]{index=3}
#         tc["_item_id"] = item_id                 # internal
#         tc["id"] = item.get("call_id") or tc.get("id")
#         tc["type"] = "function"
#         fn = tc.setdefault("function", {"name": None, "arguments": ""})
#         if item.get("name") is not None:
#             fn["name"] = item["name"]
#         if item.get("arguments") is not None:
#             fn["arguments"] = item["arguments"] or ""

#         # Might already be complete
#         now_complete = bool(fn.get("name")) and _safe_json_loads(fn.get("arguments") or "")
#         tc["_complete"] = now_complete
#         if now_complete:
#             new_built = tc

#         return tool_calls, new_built

#     # 2) Arguments stream in deltas
#     if t == "response.function_call_arguments.delta":
#         item_id = event.get("item_id")
#         if not item_id:
#             return tool_calls, None

#         tc = ensure_item_slot(item_id)
#         fn = tc.setdefault("function", {"name": None, "arguments": ""})

#         delta_args = event.get("delta") or ""
#         fn["arguments"] = (fn.get("arguments") or "") + delta_args  # append fragment :contentReference[oaicite:4]{index=4}

#         # detect built transition
#         name = fn.get("name")
#         args = fn.get("arguments") or ""
#         was_complete = bool(tc.get("_complete"))
#         now_complete = bool(name) and _safe_json_loads(args)
#         tc["_complete"] = now_complete
#         if now_complete and not was_complete:
#             new_built = tc

#         return tool_calls, new_built

#     # 3) Arguments finalize
#     if t == "response.function_call_arguments.done":
#         item_id = event.get("item_id")
#         if not item_id:
#             return tool_calls, None

#         tc = ensure_item_slot(item_id)
#         fn = tc.setdefault("function", {"name": None, "arguments": ""})

#         final_args = event.get("arguments")
#         if final_args is not None:
#             fn["arguments"] = final_args

#         name = fn.get("name")
#         args = fn.get("arguments") or ""
#         now_complete = bool(name) and _safe_json_loads(args)
#         tc["_complete"] = now_complete
#         if now_complete:
#             new_built = tc

#         return tool_calls, new_built

#     # Optional: if you also want to support "response.output_item.done" updates for function_call items.
#     if t == "response.output_item.done":
#         item = event.get("item") or {}
#         if item.get("type") != "function_call":
#             return tool_calls, None

#         item_id = item.get("id")
#         if not item_id:
#             return tool_calls, None

#         tc = ensure_item_slot(item_id)
#         tc["_item_id"] = item_id
#         tc["id"] = item.get("call_id") or tc.get("id")

#         fn = tc.setdefault("function", {"name": None, "arguments": ""})
#         if item.get("name") is not None:
#             fn["name"] = item["name"]
#         if item.get("arguments") is not None:
#             fn["arguments"] = item["arguments"] or ""

#         now_complete = bool(fn.get("name")) and _safe_json_loads(fn.get("arguments") or "")
#         tc["_complete"] = now_complete
#         if now_complete:
#             new_built = tc

#         return tool_calls, new_built

#     return tool_calls, None


# def build_tool_calls(cur_message: Message, delta: Dict[str, Any]) -> Tuple[List[ToolCall], Optional[ToolCall]]:
#     """
#     Dispatcher that accepts either:
#       - Chat Completions per-choice delta (chunk["choices"][i]["delta"])
#       - Responses streaming event dict (event["type"] starts with "response.")
#       - Or a full Chat Completions chunk (object == "chat.completion.chunk") (best-effort)

#     Returns: (all_tool_calls_so_far, newly_built_tool_call_or_None)
#     """
#     # If they accidentally pass a full chat chunk, peel out delta (choice 0).
#     if delta.get("object") == "chat.completion.chunk" and "choices" in delta:
#         choices = delta.get("choices") or []
#         if choices and isinstance(choices[0], dict):
#             delta = (choices[0].get("delta") or {})

#     # Responses streaming events are keyed by "type" like "response.function_call_arguments.delta". :contentReference[oaicite:5]{index=5}
#     if isinstance(delta.get("type"), str) and delta["type"].startswith("response."):
#         return build_tool_calls_from_responses_stream_event(cur_message, delta)

#     # Otherwise treat as Chat Completions delta.
#     return build_tool_calls_from_chat_completions_delta(cur_message, delta)
# ```

# ### Notes that matter in real apps

# * **Responses API**: the identifier you generally need to send tool outputs back is `call_id` (mapped above to `tool_call["id"]`). The `function_call` output item contains it in the non-streamed response. ([GitHub][1])
# * **Chat Completions API**: streaming uses `index` inside each tool-call delta to let you stitch multiple concurrent tool calls in order; you must **append** `function.arguments` fragments.
# * In both APIs, it’s normal for `arguments` to be invalid JSON until the last fragment arrives, hence the `_safe_json_loads()` “built” test.

# If you want, I can also provide a small “driver” example showing how to feed this from an SSE loop (Responses) vs an iterator of `chat.completion.chunk` objects (Chat Completions), but the core reconstruction logic above is the tricky part.

# [1]: https://raw.githubusercontent.com/openai/openai-openapi/refs/heads/manual_spec/openapi.yaml "raw.githubusercontent.com"


# Here’s a detailed breakdown of the structure of a response from OpenAI Chat Completions API — covering both non-streaming (standard) and streaming modes. I include the major fields, typical shapes, and how incremental updates are delivered during streaming.

# Sources: official OpenAI chat docs and streaming-chunk specification. ([OpenAI Platform][1])

# ---

# ## 1. Non-streaming Chat Completion response (stream=false)

# When you call POST `/v1/chat/completions` (or corresponding SDK method) with `stream=false` or without `stream`, you receive a JSON response similar to:

# ```jsonc
# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion",
#   "created": 1694268000,
#   "model": "gpt-4o-mini",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hello, how can I help you today?"
#       },
#       "finish_reason": "stop"
#     }
#     /* possibly more choices if n>1 */
#   ],
#   "usage": {
#     "prompt_tokens": 10,
#     "completion_tokens": 15,
#     "total_tokens": 25
#   }
# }
# ```

# ### Fields explained

# * `id` (string): Unique identifier for the chat completion request/response. ([OpenAI Platform][2])
# * `object` (string): The type of object — `"chat.completion"`. ([OpenAI Platform][2])
# * `created` (integer): Unix timestamp (in seconds) indicating when this completion was created. ([OpenAI Platform][2])
# * `model` (string): The name of the model used. ([OpenAI Platform][1])
# * `choices` (array): A list of “completion alternatives.” Usually `n=1`, so one element. Each choice includes:

#   * `index` (integer): position among choices (starting at 0). ([OpenAI Platform][1])
#   * `message` (object): the assistant’s reply, with:

#     * `role` (string): typically `"assistant"`. ([OpenAI Platform][1])
#     * `content` (string): the full text generated by the model. ([OpenAI Platform][1])
#     * *(In more advanced uses, there may be other fields inside message: e.g. for function-calling or structured outputs — though the canonical minimal spec shows only role + content.)*
#   * `finish_reason` (string or null): why generation stopped. Common values: `"stop"`, `"length"`, etc. ([OpenAI Platform][1])
# * `usage` (object): token usage information:

#   * `prompt_tokens` (integer): number of tokens in the prompt (messages you sent). ([OpenAI Platform][1])
#   * `completion_tokens` (integer): number of tokens in the generated completion. ([OpenAI Platform][1])
#   * `total_tokens` (integer): sum of prompt + completion tokens. ([OpenAI Platform][1])

# That is the core of a typical non-streaming response.

# If you use features like structured output or function calling (depending on the SDK / parameters), there may be additional fields (e.g. a `function_call` inside `message` instead of `content`). The docs for structured outputs mention that you can supply a `response_format` parameter (e.g. JSON schema) — though that's more clearly documented in the context of newer “Responses API.” ([豆蔵デベロッパーサイト][3])

# ---

# ## 2. Streaming Chat Completion response (stream=true)

# If you set `stream = true` in the request, then instead of a single JSON response, the API returns a **stream of chunks**, usually via server-sent events (SSE). Each chunk is a JSON object with schema defined for streaming mode. ([OpenAI Platform][2])

# A streaming chunk looks like:

# ```jsonc
# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion.chunk",
#   "created": 1694268000,
#   "model": "gpt-4o-mini",
#   "choices": [
#     {
#       "index": 0,
#       "delta": {
#         "role": "assistant",
#         "content": "Hello"
#       },
#       "finish_reason": null,
#       "logprobs": null
#     }
#     /* possibly more choices if n>1 */
#   ]
#   /* optionally usage if streaming with usage included */
# }
# ```

# ### Streaming-chunk fields explained

# * `id`, `object`, `created`, `model`: same semantics as non-streaming, but `object` is `"chat.completion.chunk"`. ([OpenAI Platform][2])
# * `choices` (array): similar to non-streaming, but each element has:

#   * `index` (integer): as before. ([OpenAI Platform][2])
#   * `delta` (object): represents a **partial update** from the model for that choice. The delta may contain one or more of:

#     * `role`: e.g. `"assistant"` (typically only in the first chunk). ([nikkie-ftnextの日記][4])
#     * `content`: a string fragment (often a token or partial token) of the assistant’s message. For example, first chunk may have `content=""`, then subsequent chunks carry the actual text. ([Zenn][5])
#   * `finish_reason`: null until the final chunk; on final chunk it might be `"stop"`, `"length"`, etc. ([OpenAI Platform][2])
#   * `logprobs`: often null (used if you request token-level log-probabilities). ([OpenAI Platform][2])
# * Optionally, some streaming configurations allow **usage info** to be included at the end (depending on `stream_options: { "include_usage": true }`). ([Zenn][5])

# ### How to reconstruct full response from chunks

# When streaming:

# * The first chunk may only set `delta.role`, with empty `content`. Typically indicates the assistant’s role. ([nikkie-ftnextの日記][4])
# * Subsequent chunks have `delta.content` fragments — you accumulate them (concatenate in order) to build the full reply. ([nikkie-ftnextの日記][4])
# * The final chunk has a non-null `finish_reason`. Once you see that, you know the reply is done. ([OpenAI Platform][2])
# * If usage info is enabled, it may arrive in the last chunk (or a final chunk) so you know token usage. ([Zenn][5])

# Many SDKs abstract this for you: e.g. they expose an iterable stream, where each iteration yields a chunk object. You check `chunk.choices[0].delta.content` and append it until done. ([nikkie-ftnextの日記][4])

# ---

# ## 3. Comparison: Chat Completion API vs Responses API

# While you asked for Chat Completion structure, it helps to note how it differs from the newer OpenAI Responses API (which we reviewed previously):

# * Chat Completion returns a **flat choices/messages** structure (role + content), simpler by design.
# * Responses API (the newer unified interface) is more flexible: it supports not only text but also images, audio, tool-calls, function-calling, more detailed item-based tracking, and structured content. ([OpenAI Platform][6])
# * Streaming in Chat Completion is token-delta based (via `delta.content`) — primarily for incremental text output. ([OpenAI Platform][2])
# * Responses API allows richer item types (messages, tool calls, tool outputs, structured outputs) rather than just "message".

# Thus Chat Completion is simpler, but less expressive / extensible.

# ---

# ## 4. Extended / optional features in Chat Completion

# Depending on request parameters (or upcoming SDK features), Chat Completion responses may include or support:

# * **Multiple choices** (`n > 1`): you get multiple alternatives in `choices` array. You pick `choices[0]` or iterate all. ([OpenAI Platform][1])
# * **Function calling / structured outputs**: If you request output in structured format (e.g. JSON via `response_format`) — note that this is more clearly supported in the Responses API, but there are community reports of attempts with Chat Completion as well. ([豆蔵デベロッパーサイト][3])
# * **Logprobs, token-level data**: If you request them (rare), the `logprobs` field may be non-null inside choices. ([OpenAI Platform][2])
# * **Usage customization**: Some optional flags (depending on client) to include usage tokens in streaming mode. ([Zenn][5])

# ---

# ## 5. Why understanding the structure matters — and pitfalls

# Understanding this structure is important because:

# * In streaming mode you **must accumulate deltas correctly** — naive JSON parsing per chunk may fail or produce incomplete output.
# * If you rely on token counts or usage, you must ensure your `stream_options` (if available) include usage, otherwise the chunk stream may lack usage info. ([Zenn][5])
# * Mixing Chat Completion with structured output or function calling requires careful handling: while possible, many of the newer “structured outputs” features are more robustly supported via Responses API. Some users report unexpected extra whitespace or formatting quirks even when specifying strict JSON schema. ([OpenAI Developer Community][7])
# * Upgrades in backend (“system_fingerprint” changes, model updates) may subtly affect tokenization / output — but Chat Completion JSON schema remains stable.

# ---

# If you like — I can also **draw a type-definition** (in e.g. TypeScript or Python `dataclass`) for Chat Completion response (both streaming and non-streaming).
# Would you like me to build that for you now?

# [1]: https://platform.openai.com/docs/api-reference/chat?utm_source=chatgpt.com "API Reference"
# [2]: https://platform.openai.com/docs/api-reference/chat-streaming/streaming?utm_source=chatgpt.com "The chat completion chunk object"
# [3]: https://developer.mamezou-tech.com/blogs/2024/08/10/openai-structured-output-intro/?utm_source=chatgpt.com "OpenAIのStructured Outputsを使ってAIの出力スキーマを定義 ..."
# [4]: https://nikkie-ftnext.hatenablog.com/entry/openai-chat-completions-python-client-stream?utm_source=chatgpt.com "OpenAIのChat completions APIをstreamで使う（Python ..."
# [5]: https://zenn.dev/tomodo_ysys/articles/openai-streaming-token-count?utm_source=chatgpt.com "[OpenAI API] ストリーミングレスポンスの構造を理解して ..."
# [6]: https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com "Responses API reference"
# [7]: https://community.openai.com/t/api-response-is-not-json-parsable-despite-specified-response-format/1014311?utm_source=chatgpt.com "API response is not JSON parsable despite specified ..."

