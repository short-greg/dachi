# 1st party
import typing as t
from typing import Literal

# 3rd party
import pydantic
import openai

import typing as t
# local
from ..core import (
    Msg, Resp, DeltaResp, BaseDialog
)
from ..core._tool import BaseTool, ToolBuffer, ToolChunk, ToolUse
from ._ai import LangEngine
import json

# Format conversion helper functions

# ============================================================================
# Tool Call Processing Helper Functions
# ============================================================================


class OpenAIChat(LangEngine):
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

    def to_api_input(self, messages: list[dict], tools: list[BaseTool], format_override, **kwargs) -> t.Dict:
        """Convert to OpenAI Responses API input format."""
        # TODO: must confirm this is correct
        api_input = {
            "messages": messages,
            **kwargs
        }
        
        # Add tools if present
        if tools:
            api_input["tools"] = None
            
        # Add text format if present
        if isinstance(format_override, pydantic.BaseModel):
            format_override = format_override.model_dump()
        api_input['response_format'] = format_override
        
        if kwargs.get('stream', False) is True:
            api_input['stream'] = True
        
        return api_input
    
    def convert_tools(self, tools: list[BaseTool]) -> list[dict]:
        """
        Convert Dachi BaseTool objects into Chat Completions `tools` payload entries.

        We default to Structured Outputs for function calls by setting `strict: true`.
        This expects the top-level object schema to be "closed" (additionalProperties: false)
        and to explicitly list required properties. We enforce those at the top level.
        """
        if not tools:
            return []

        openai_tools: list[dict] = []

        for tool in tools:
            name = getattr(tool, "name", None)
            if not isinstance(name, str) or not name:
                continue

            description = getattr(tool, "description", "")
            if not isinstance(description, str):
                description = ""

            params: dict = {"type": "object", "properties": {}}
            input_model = getattr(tool, "input_model", None)

            if input_model is not None:
                try:
                    if hasattr(input_model, "model_json_schema"):
                        params = input_model.model_json_schema()
                    elif hasattr(input_model, "schema"):
                        params = input_model.schema()
                except Exception:
                    params = {"type": "object", "properties": {}}

            # Strict-mode-friendly top-level adjustments (best-effort, top-level only)
            if isinstance(params, dict) and params.get("type") == "object":
                props = params.get("properties")
                if isinstance(props, dict):
                    # In strict mode, OpenAI expects an explicit required list.
                    if "required" not in params:
                        params["required"] = list(props.keys())
                    # In strict mode, OpenAI expects objects to disallow extra keys.
                    if "additionalProperties" not in params:
                        params["additionalProperties"] = False

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "strict": True,
                        "parameters": params,
                    },
                }
            )

        return openai_tools

    def convert_format_override(
        self,
        format_override: t.Union[bool, str, dict, t.Type[pydantic.BaseModel], pydantic.BaseModel, None],
    ) -> t.Optional[dict]:
        """
        Convert Dachi format_override into Chat Completions `response_format`.

        Returns either:
        - None (no override)
        - {"type":"json_object"} for JSON mode
        - {"type":"json_schema","json_schema":{...}} for Structured Outputs
        """
        if format_override is None or format_override is False:
            return None

        # Simple JSON mode
        if format_override is True or format_override == "json":
            return {"type": "json_object"}

        # Default text mode: omit response_format (Chat Completions default is text)
        if format_override == "text":
            return None

        # If caller already provided a response_format dict, pass through
        if isinstance(format_override, dict):
            if "type" in format_override:
                return format_override
            # Otherwise interpret as a raw JSON Schema to wrap as structured output
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": format_override,
                },
            }

        # Pydantic model class or instance -> JSON Schema -> structured output wrapper
        model_cls: t.Optional[t.Type[pydantic.BaseModel]] = None
        if isinstance(format_override, type) and issubclass(format_override, pydantic.BaseModel):
            model_cls = format_override
        elif isinstance(format_override, pydantic.BaseModel):
            model_cls = format_override.__class__

        if model_cls is not None:
            if hasattr(model_cls, "model_json_schema"):
                schema = model_cls.model_json_schema()
            else:
                schema = model_cls.schema()  # pydantic v1 fallback

            return {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(model_cls, "__name__", "response"),
                    "strict": True,
                    "schema": schema,
                },
            }

        return None
        
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

    def set_core_elements(self, resp: "Resp", message: dict):
        """
        Chat Completions (non-streamed) core extraction:
        - id/model/usage from payload
        - role/content from choices[0].message
        - finish_reason/logprobs from choices[0]
        """
        payload = message

        resp.raw = payload
        resp.id = payload.get("id")
        resp.model = payload.get("model")
        resp.usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        resp.choices = payload.get("choices") if isinstance(payload.get("choices"), list) else None

        choices = payload.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}

        resp.finish_reason = choice0.get("finish_reason")
        resp.logprobs = choice0.get("logprobs")

        msg = choice0.get("message") or {}
        resp.role = msg.get("role", "assistant")

        content = msg.get("content", "")
        if isinstance(content, str):
            resp.text = content
        elif isinstance(content, list):
            # Join text blocks if assistant content is in parts form
            out_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out_parts.append(part["text"])
                elif isinstance(part.get("text"), str):
                    out_parts.append(part["text"])
            resp.text = "".join(out_parts)
        else:
            resp.text = ""

    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # full chat.completion.chunk payload
    ) -> None:
        # --- delta: reset to "this chunk only"
        cur_delta.text = None
        cur_delta.thinking = None
        cur_delta.citations = None
        cur_delta.tool = None
        cur_delta.usage = None
        cur_delta.finish_reason = None

        # Defensive: allow prev_resp to be empty-ish
        if prev_resp is None:
            prev_resp = Resp()

        # --- extract first choice delta (Chat chunks are choice-based)
        choices = delta_message.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            # Still propagate stable metadata if present
            mid = delta_message.get("id")
            model = delta_message.get("model")

            cur_resp.id = prev_resp.id or (mid if isinstance(mid, str) else None)
            cur_resp.model = prev_resp.model or (model if isinstance(model, str) else None)
            cur_resp.role = prev_resp.role or "assistant"
            cur_resp.text = prev_resp.text or ""
            cur_resp.finish_reason = prev_resp.finish_reason
            cur_resp.usage = prev_resp.usage or {}
            cur_resp.raw = delta_message
            return

        choice0 = choices[0]
        delta = choice0.get("delta")
        delta = delta if isinstance(delta, dict) else {}

        text_inc = delta.get("content")               # incremental text fragment
        role_inc = delta.get("role")                  # may appear only once (first chunk)
        finish_inc = choice0.get("finish_reason")     # usually only last chunk

        usage_inc = delta_message.get("usage")        # only if stream_options include it
        usage_inc = usage_inc if isinstance(usage_inc, dict) else None

        # --- accumulate into cur_resp (prev + increment)
        mid = delta_message.get("id")
        model = delta_message.get("model")
        cur_resp.id = prev_resp.id or (mid if isinstance(mid, str) else None)
        cur_resp.model = prev_resp.model or (model if isinstance(model, str) else None)

        cur_resp.role = (
            role_inc if isinstance(role_inc, str) and role_inc else (prev_resp.role or "assistant")
        )

        prev_text = prev_resp.text if isinstance(prev_resp.text, str) else (prev_resp.text or "")
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

        # keep raw around
        cur_resp.raw = delta_message
            
    def set_tools(self, resp: "Resp", message: dict):
        """
        Chat Completions (non-streamed):
        payload["choices"][0]["message"]["tool_calls"] -> resp.tool_use
        """
        resp.tool_use = []

        tool_defs = (
            getattr(self, "_tool_defs", None)
            or getattr(self, "_tools", None)
            or getattr(self, "tools", None)
            or []
        )
        if not isinstance(tool_defs, list) or not tool_defs:
            return

        tool_map = {}
        for tool in tool_defs:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                tool_map[name] = tool

        choices = message.get("choices") or []
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
            return

        msg = (choices[0].get("message") or {})
        if not isinstance(msg, dict):
            return

        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            return

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            fn = tool_call.get("function") or {}
            if not isinstance(fn, dict):
                continue

            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue

            tool_def = tool_map.get(name)
            if tool_def is None:
                continue

            args_text = fn.get("arguments", "")
            if not isinstance(args_text, str):
                args_text = ""

            args = {}
            if args_text.strip():
                try:
                    parsed = json.loads(args_text)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}

            tool_id = tool_call.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                tool_id = f"tool_{len(resp.tool_use)}"

            tool_use = tool_def.to_tool_call(tool_id=tool_id, **args)

            provider_id = tool_call.get("id")
            if isinstance(provider_id, str) and provider_id:
                tool_use.id = provider_id

            resp.tool_use.append(tool_use)

    def set_tools_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # chat.completion.chunk payload
    ):
        """
        Chat Completions streaming:
        chunk["choices"][0]["delta"]["tool_calls"][...]

        Contract:
        - cur_delta.tool stays None unless a tool call fully completes in this chunk.
        - newly completed ToolUse objects get appended to cur_resp.tool_use
        - accumulation is cleared once completion happens (no lingering partials).
        """
        cur_delta.tool = None
        cur_resp.tool_use = list(prev_resp.tool_use)

        if not isinstance(getattr(cur_delta, "accumulation", None), dict):
            cur_delta.accumulation = {}

        buf = cur_delta.accumulation.get("tool_buffer")
        if buf is None:
            tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
            buf = ToolBuffer(tools=tools)
            cur_delta.accumulation["tool_buffer"] = buf
            cur_delta.accumulation["tool_buffer_emitted"] = 0
            cur_delta.accumulation["tool_call_meta"] = {}

        emitted_n = cur_delta.accumulation.get("tool_buffer_emitted", 0)
        if not isinstance(emitted_n, int):
            emitted_n = 0
            cur_delta.accumulation["tool_buffer_emitted"] = 0

        meta = cur_delta.accumulation.get("tool_call_meta")
        if not isinstance(meta, dict):
            meta = {}
            cur_delta.accumulation["tool_call_meta"] = meta

        choices = delta_message.get("choices") or []
        choice0 = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish_reason = choice0.get("finish_reason")

        delta = choice0.get("delta")
        delta = delta if isinstance(delta, dict) else {}

        tool_calls_delta = delta.get("tool_calls")

        # Back-compat: deprecated delta["function_call"] -> tool_calls[0]
        if not isinstance(tool_calls_delta, list) or not tool_calls_delta:
            fc = delta.get("function_call")
            if isinstance(fc, dict):
                tool_calls_delta = [{
                    "index": 0,
                    "id": None,
                    "type": "function",
                    "function": fc,
                }]
            else:
                tool_calls_delta = []

        # 1) Accumulate chunks
        for tc in tool_calls_delta:
            if not isinstance(tc, dict):
                continue

            call_index = tc.get("index")
            if not isinstance(call_index, int):
                continue

            fn = tc.get("function")
            fn = fn if isinstance(fn, dict) else {}

            tc_id = tc.get("id") if isinstance(tc.get("id"), str) else None
            fn_name = fn.get("name") if isinstance(fn.get("name"), str) else None
            fn_args = fn.get("arguments") if isinstance(fn.get("arguments"), str) else None

            existing = meta.get(call_index)
            existing = existing if isinstance(existing, dict) else {}

            # IMPORTANT: ToolBuffer keying prefers id if present; if OpenAI only sends id once,
            # we must remember it and pass it on every chunk afterwards.
            stable_id = tc_id or (existing.get("id") if isinstance(existing.get("id"), str) else None)
            stable_name = fn_name or (existing.get("name") if isinstance(existing.get("name"), str) else None)

            meta[call_index] = {"id": stable_id, "name": stable_name}

            try:
                buf.append(
                    ToolChunk(
                        id=stable_id,
                        turn_index=0,  # chat doesn't expose a message index; stable constant
                        call_index=call_index,
                        name=stable_name,
                        args_text_delta=fn_args,
                        done=False,
                    )
                )
            except Exception:
                # Don't crash streaming on partial tool-call issues; subclass can inspect raw if desired.
                pass

        # 2) Finalize on terminal chunk even if it contains no delta.tool_calls
        if finish_reason == "tool_calls":
            for call_index, rec in list(meta.items()):
                if not isinstance(call_index, int) or not isinstance(rec, dict):
                    continue

                stable_id = rec.get("id") if isinstance(rec.get("id"), str) else None
                stable_name = rec.get("name") if isinstance(rec.get("name"), str) else None

                try:
                    buf.append(
                        ToolChunk(
                            id=stable_id,
                            turn_index=0,
                            call_index=call_index,
                            name=stable_name,
                            args_text_delta=None,
                            done=True,
                        )
                    )
                except Exception:
                    pass

        # 3) Emit newly completed calls
        new_calls = buf._calls[emitted_n:]
        if new_calls:
            cur_resp.tool_use.extend(new_calls)
            cur_delta.accumulation["tool_buffer_emitted"] = emitted_n + len(new_calls)

            last = new_calls[-1]
            try:
                cur_delta.tool = json.dumps(last.inputs.model_dump())
            except Exception:
                cur_delta.tool = "{}"

        # 4) Clear accumulation when tool-calling turn is finished (no lingering partials)
        if finish_reason == "tool_calls":
            cur_delta.accumulation.pop("tool_buffer", None)
            cur_delta.accumulation.pop("tool_buffer_emitted", None)
            cur_delta.accumulation.pop("tool_call_meta", None)


    def convert_messages(self, msg: Msg | BaseDialog | str) -> list[dict]:
        """Convert Dachi Msg/BaseDialog into Chat Completions `messages` list (no tools)."""
        if isinstance(msg, str):
            msg = Msg(role="user", text=msg)

        original_messages = [msg] if isinstance(msg, Msg) else list(msg)
        openai_messages: list[dict] = []

        for m in original_messages:
            # --- coerce text to a string
            if isinstance(m.text, str):
                text_content = m.text
            elif m.text is None:
                text_content = ""
            else:
                text_content = str(m.text)

            openai_msg: dict[str, t.Any] = {"role": m.role}

            # Optional alias/name (harmless for most roles; omit if you dislike it)
            alias = getattr(m, "alias", None)
            if isinstance(alias, str) and alias:
                openai_msg["name"] = alias

            attachments = getattr(m, "attachments", None) or []
            if attachments:
                parts: list[dict] = []

                if text_content:
                    parts.append({"type": "text", "text": text_content})

                for att in attachments:
                    if getattr(att, "kind", None) != "image":
                        continue

                    data = getattr(att, "data", None)
                    if not isinstance(data, str) or not data:
                        continue

                    image_url = data
                    if not image_url.startswith("data:"):
                        mime = getattr(att, "mime", None) or "image/png"
                        image_url = f"data:{mime};base64,{image_url}"

                    parts.append({"type": "image_url", "image_url": {"url": image_url}})

                # If we built blocks, use block form; otherwise fall back to plain text
                openai_msg["content"] = parts if parts else text_content
            else:
                openai_msg["content"] = text_content

            openai_messages.append(openai_msg)

        return openai_messages

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


class OpenAIResp(LangEngine):
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

    def to_api_input(
        self,
        messages: list[dict],
        tools: list[dict],
        format_override: t.Union[bool, dict, None],
        **kwargs,
    ) -> t.Dict:
        """Build payload for OpenAI Responses API."""
        api_input: dict[str, t.Any] = {"input": messages}
        api_input.update(kwargs)

        if tools:
            api_input["tools"] = tools

        if format_override:
            # Responses API uses `text.format` for structured outputs in this codebase style
            # (caller passes converted dict, e.g. {"type":"json_schema", ...})
            api_input.setdefault("text", {})
            if isinstance(api_input["text"], dict):
                api_input["text"]["format"] = format_override

        return api_input
    

    def convert_tools(self, tools: list[BaseTool]) -> list[dict]:
        """Convert Dachi BaseTool objects into Responses API `tools` entries."""
        if not tools:
            return []

        out: list[dict] = []

        for tool in tools:
            name = getattr(tool, "name", None)
            if not isinstance(name, str) or not name:
                continue

            description = getattr(tool, "description", "")
            if not isinstance(description, str):
                description = ""

            # Build JSON Schema for tool parameters
            params: dict = {"type": "object", "properties": {}}
            input_model = getattr(tool, "input_model", None)
            if input_model is not None:
                try:
                    if hasattr(input_model, "model_json_schema"):
                        params = input_model.model_json_schema()
                    elif hasattr(input_model, "schema"):
                        params = input_model.schema()
                except Exception:
                    params = {"type": "object", "properties": {}}

            out.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": params,
                    "strict": True,
                }
            )

        return out

    def convert_messages(self, msg: Msg | BaseDialog | str) -> list[dict]:
        """Convert Dachi Msg/BaseDialog into Responses API `input` items (role + content blocks)."""
        if isinstance(msg, str):
            msg = Msg(role="user", text=msg)

        original_messages = [msg] if isinstance(msg, Msg) else list(msg)

        openai_messages: list[dict] = []
        for m in original_messages:
            text_content: str
            if isinstance(m.text, str):
                text_content = m.text
            elif m.text is None:
                text_content = ""
            else:
                text_content = str(m.text)

            content: list[dict] = []
            if text_content:
                content.append({"type": "input_text", "text": text_content})

            for att in (getattr(m, "attachments", None) or []):
                if getattr(att, "kind", None) != "image":
                    continue

                image_url = getattr(att, "data", "")
                if not isinstance(image_url, str) or not image_url:
                    continue

                # If it's not already a URL or a data: URI, treat as base64 and wrap it.
                if not (image_url.startswith("data:") or image_url.startswith("http://") or image_url.startswith("https://")):
                    mime = getattr(att, "mime", None) or "image/png"
                    image_url = f"data:{mime};base64,{image_url}"

                content.append({"type": "input_image", "image_url": image_url})

            # Responses API expects content blocks; keep it non-empty.
            if not content:
                content = [{"type": "input_text", "text": ""}]

            openai_messages.append({"role": m.role, "content": content})

        return openai_messages

    def _responses_text_from_message_item(self, msg_item: t.Dict[str, t.Any]) -> str:
        content = msg_item.get("content", [])
        if not isinstance(content, list):
            return ""
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )

    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ):
        # --- delta: reset to "this chunk only"
        cur_delta.text = None
        cur_delta.usage = None
        cur_delta.finish_reason = None  # Responses events don't map cleanly; leave unset here

        # --- tolerate first chunk where prev_resp may be None
        prev_id = getattr(prev_resp, "id", None) if prev_resp is not None else None
        prev_model = getattr(prev_resp, "model", None) if prev_resp is not None else None
        prev_role = getattr(prev_resp, "role", None) if prev_resp is not None else None
        prev_text = getattr(prev_resp, "text", "") if prev_resp is not None else ""
        prev_usage = getattr(prev_resp, "usage", {}) if prev_resp is not None else {}

        if not isinstance(prev_text, str):
            prev_text = ""
        if not isinstance(prev_usage, dict):
            prev_usage = {}

        ev_type = delta_message.get("type")

        resp_obj = delta_message.get("response")
        resp_obj = resp_obj if isinstance(resp_obj, dict) else {}

        # Identity/model can arrive on created/completed events via resp_obj, or per-event response_id
        response_id = delta_message.get("response_id")
        if not isinstance(response_id, str):
            response_id = None

        cur_resp.id = prev_id or response_id or (resp_obj.get("id") if isinstance(resp_obj.get("id"), str) else None)
        cur_resp.model = prev_model or (resp_obj.get("model") if isinstance(resp_obj.get("model"), str) else None)

        # Role is typically not included on text delta events; keep previous/default
        cur_resp.role = prev_role or "assistant"

        # Text is typically delivered via response.output_text.delta
        if ev_type == "response.output_text.delta":
            text_inc = delta_message.get("delta")
            if isinstance(text_inc, str) and text_inc:
                cur_delta.text = text_inc
                cur_resp.text = prev_text + text_inc
            else:
                cur_resp.text = prev_text
        else:
            cur_resp.text = prev_text

        # Usage commonly arrives on/near completion in resp_obj["usage"] (treat as latest total)
        usage_inc = resp_obj.get("usage")
        usage_inc = usage_inc if isinstance(usage_inc, dict) else None

        if usage_inc is not None:
            cur_delta.usage = usage_inc
            cur_resp.usage = usage_inc
        else:
            cur_resp.usage = prev_usage

        # Keep raw event around for debugging
        cur_resp.raw = delta_message

    def set_tools(self, resp: "Resp", message: dict) -> None:
        """
        Responses API (non-streamed):

        The full response payload has an `output` array of items, where tool calls
        are typically objects with:
            {
            "type": "function_call" | "tool_call",
            "call_id": "...",          # id the model uses for this call
            "name": "...",             # tool/function name
            "arguments": "{...json...}"
            ...
            }

        This method converts those into ToolUse objects on resp.tool_use.
        """
        # Resolve available tools -> map by name
        tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
        if not isinstance(tools, list) or not tools:
            resp.tool_use = []
            return

        tool_map: dict[str, BaseTool] = {}
        for tool in tools:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                tool_map[name] = tool

        output = message.get("output") or []
        if not isinstance(output, list) or not output:
            resp.tool_use = []
            return

        tool_uses: list[ToolUse] = []
        unknown_calls: list[dict] = []

        for item in output:
            if not isinstance(item, dict):
                continue

            itype = item.get("type")
            if itype not in ("function_call", "tool_call"):
                continue

            # Common Responses shape: {"type":"function_call","call_id":...,"name":...,"arguments":...}
            name = item.get("name") or item.get("tool_name")
            args_text = item.get("arguments", "")
            call_id = item.get("call_id") or item.get("id")

            if not isinstance(name, str) or not isinstance(args_text, str):
                continue

            tool_def = tool_map.get(name)
            if tool_def is None:
                unknown_calls.append(item)
                continue

            # Parse arguments JSON (best-effort)
            args: dict = {}
            if args_text.strip():
                try:
                    parsed = json.loads(args_text)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}

            tool_id = call_id if isinstance(call_id, str) and call_id else f"tool_{len(tool_uses)}"

            try:
                tool_use = tool_def.to_tool_call(tool_id=tool_id, **args)
            except Exception:
                # If something goes wrong building ToolUse, treat as unknown
                unknown_calls.append(item)
                continue

            # Preserve provider call id on ToolUse.id if present
            if isinstance(call_id, str) and call_id:
                tool_use.id = call_id

            tool_uses.append(tool_use)

        resp.tool_use = tool_uses

        if unknown_calls:
            if not isinstance(resp.meta, dict):
                resp.meta = {}
            resp.meta.setdefault("unknown_tool_calls", [])
            if isinstance(resp.meta["unknown_tool_calls"], list):
                resp.meta["unknown_tool_calls"].extend(unknown_calls)

    def convert_format_override(
        self,
        format_override: t.Union[bool, dict, pydantic.BaseModel, t.Type[pydantic.BaseModel], None],
    ) -> t.Union[bool, dict, None]:
        """Convert Dachi format override to Responses-API-compatible `text.format` payload."""
        if format_override is None or format_override is False:
            return None

        # "json mode" (no schema)
        if format_override is True:
            return {"type": "json_object"}

        # Already-converted / user-supplied format dict
        if isinstance(format_override, dict):
            return format_override

        model_cls: t.Optional[t.Type[pydantic.BaseModel]] = None

        # Accept either a BaseModel instance or a BaseModel subclass
        if isinstance(format_override, pydantic.BaseModel):
            model_cls = format_override.__class__
        elif isinstance(format_override, type) and issubclass(format_override, pydantic.BaseModel):
            model_cls = format_override

        if model_cls is None:
            return None

        # Build JSON Schema
        schema: dict
        try:
            schema = model_cls.model_json_schema()  # pydantic v2
        except Exception:
            try:
                schema = model_cls.schema()  # pydantic v1
            except Exception:
                schema = {}

        return {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        }


    def set_tools_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ) -> dict:
        """
        Responses API (streaming events):
        - Tracks tool call items by item_id
        - Accumulates args text fragments into a ToolBuffer
        - Emits ToolUse objects only when a call completes (done=True)
        """
        cur_delta.tool = None
        cur_resp.tool_use = list(prev_resp.tool_use)

        # --- persistent state across chunks (store on resp.meta so it survives via prev_resp)
        if not isinstance(getattr(cur_resp, "meta", None), dict):
            cur_resp.meta = {}
        if prev_resp is not None and isinstance(getattr(prev_resp, "meta", None), dict):
            # keep any existing meta; but prefer to carry tool-stream state forward
            if "_responses_tool_stream" in prev_resp.meta and "_responses_tool_stream" not in cur_resp.meta:
                cur_resp.meta["_responses_tool_stream"] = prev_resp.meta["_responses_tool_stream"]

        state = cur_resp.meta.get("_responses_tool_stream")
        if not isinstance(state, dict):
            tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
            state = {
                "buf": ToolBuffer(tools=tools),
                "emitted": 0,
                "by_item_id": {},      # item_id -> {"id": call_id|None, "name": str|None, "call_index": int, "has_text": bool}
                "next_call_index": 0,
            }
            cur_resp.meta["_responses_tool_stream"] = state

        buf = state.get("buf")
        if not isinstance(buf, ToolBuffer):
            tools = getattr(self, "tools", None) or getattr(self, "_tools", None) or []
            buf = ToolBuffer(tools=tools)
            state["buf"] = buf

        emitted = state.get("emitted", 0)
        if not isinstance(emitted, int):
            emitted = 0
            state["emitted"] = 0

        by_item_id = state.get("by_item_id")
        if not isinstance(by_item_id, dict):
            by_item_id = {}
            state["by_item_id"] = by_item_id

        next_call_index = state.get("next_call_index", 0)
        if not isinstance(next_call_index, int):
            next_call_index = 0
            state["next_call_index"] = 0

        ev_type = delta_message.get("type")

        # Helper: get or create routing record for an item_id
        def _get_rec(item_id: str) -> dict:
            nonlocal next_call_index
            rec = by_item_id.get(item_id)
            if not isinstance(rec, dict):
                rec = {"id": None, "name": None, "call_index": next_call_index, "has_text": False}
                by_item_id[item_id] = rec
                next_call_index += 1
                state["next_call_index"] = next_call_index
            return rec

    # --- handle Responses tool-call events
    if ev_type in ("response.output_item.added", "response.output_item.done"):
        item = delta_message.get("item")
        if isinstance(item, dict) and item.get("type") in ("function_call", "tool_call"):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                rec = _get_rec(item_id)

                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id:
                    rec["id"] = call_id

                name = item.get("name") or item.get("tool_name")
                if isinstance(name, str) and name:
                    rec["name"] = name

                args_text = item.get("arguments")
                if isinstance(args_text, str) and args_text:
                    rec["has_text"] = True
                    try:
                        buf.append(
                            ToolChunk(
                                id=rec["id"],
                                turn_index=0,
                                call_index=rec["call_index"],
                                name=rec["name"],
                                args_text_delta=args_text,
                                done=False,
                            )
                        )
                    except Exception:
                        pass

                # If the provider marks the output item as done, finalize the call.
                if ev_type == "response.output_item.done":
                    try:
                        buf.append(
                            ToolChunk(
                                id=rec["id"],
                                turn_index=0,
                                call_index=rec["call_index"],
                                name=rec["name"],
                                args_text_delta=None,
                                done=True,
                            )
                        )
                    except Exception:
                        pass
                    # No further argument deltas should arrive for this item_id
                    by_item_id.pop(item_id, None)

    elif ev_type == "response.function_call_arguments.delta":
        item_id = delta_message.get("item_id")
        if isinstance(item_id, str) and item_id:
            rec = _get_rec(item_id)
            delta_txt = delta_message.get("delta")
            if isinstance(delta_txt, str) and delta_txt:
                rec["has_text"] = True
                try:
                    buf.append(
                        ToolChunk(
                            id=rec["id"],
                            turn_index=0,
                            call_index=rec["call_index"],
                            name=rec["name"],
                            args_text_delta=delta_txt,
                            done=False,
                        )
                    )
                except Exception:
                    pass

    elif ev_type == "response.function_call_arguments.done":
        item_id = delta_message.get("item_id")
        if isinstance(item_id, str) and item_id:
            rec = _get_rec(item_id)

            final_args = delta_message.get("arguments")
            if isinstance(final_args, str) and final_args and not rec.get("has_text", False):
                # If we never saw deltas, treat `arguments` as the full payload once.
                rec["has_text"] = True
                try:
                    buf.append(
                        ToolChunk(
                            id=rec["id"],
                            turn_index=0,
                            call_index=rec["call_index"],
                            name=rec["name"],
                            args_text_delta=final_args,
                            done=False,
                        )
                    )
                except Exception:
                    pass

            # Mark completion
            try:
                buf.append(
                    ToolChunk(
                        id=rec["id"],
                        turn_index=0,
                        call_index=rec["call_index"],
                        name=rec["name"],
                        args_text_delta=None,
                        done=True,
                    )
                )
            except Exception:
                pass

            by_item_id.pop(item_id, None)

    # If the response is fully completed, clear tool-stream state to avoid leakage
    if ev_type in ("response.completed", "response.failed", "response.cancelled"):
        cur_resp.meta.pop("_responses_tool_stream", None)

    # --- emit newly completed ToolUse calls from ToolBuffer
    calls = getattr(buf, "_calls", [])
    if isinstance(calls, list) and emitted < len(calls):
        new_calls = calls[emitted:]
        for c in new_calls:
            # ToolBuffer sets tool_id from provider id; mirror it into `id` for consistency
            if getattr(c, "id", None) is None and isinstance(getattr(c, "tool_id", None), str):
                c.id = c.tool_id

        cur_resp.tool_use.extend(new_calls)
        state["emitted"] = len(calls)

        # Only set delta.tool when a call completed this chunk
        last = new_calls[-1]
        try:
            cur_delta.tool = json.dumps(last.inputs.model_dump())
        except Exception:
            cur_delta.tool = "{}"
        return {"completed": new_calls}



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
