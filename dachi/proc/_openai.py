# 1st party
import typing as t

# 3rd party
import pydantic
import openai

# local
from ..core import (
    Msg, Resp, BaseModule, RespDelta, BaseDialog
)
from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ._ai import AIAdapt, LLM


class OpenAIChat(LLM, AIAdapt):
    """
    Adapter for OpenAI Chat Completions API.
    
    Converts between Dachi's unified message format and OpenAI Chat Completions API format.
    
    Message Conversion:
    - Msg.role -> message.role (user/assistant/system/tool)
    - Msg.text -> message.content 
    - Msg.attachments -> message.content (for vision/multimodal)
    - Msg.tool_calls -> role="tool" messages with tool_call_id
    
    Streaming Pattern:
    1. Accumulates text in resp.msg.text (complete message state)
    2. Sets resp.delta.text to chunk content only  
    3. Uses resp.spawn() to create next chunk response
    4. Processors use resp.out_store for stateful accumulation
    
    Unified kwargs (converted):
    - temperature, max_tokens, top_p, frequency_penalty, presence_penalty
    - stream, stop, seed, user
    
    API-specific kwargs (passed through):
    - tools, tool_choice, response_format, logprobs, top_logprobs
    - parallel_tool_calls, service_tier, stream_options
    """
    
    url: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # Create the OpenAI clients
        client_kwargs = {}
        if self.url:
            client_kwargs['base_url'] = self.url
        
        # Only create clients if we have an API key or explicit test mode
        try:
            self.client = openai.Client(**client_kwargs)
            self.async_client = openai.AsyncClient(**client_kwargs)
        except openai.OpenAIError:
            # For testing purposes, set clients to None
            self.client = None
            self.async_client = None

    def to_input(self, inp: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi format to Chat Completions format."""
        if isinstance(inp, Msg):
            messages = [self._convert_message(inp)]
        else:
            messages = [self._convert_message(msg) for msg in inp]
        
        # Add tool messages from ToolUse objects
        for msg in (inp if isinstance(inp, BaseDialog) else [inp]):
            for tool_out in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "content": str(tool_out.result),
                    "tool_call_id": tool_out.tool_call_id
                })
        
        return {
            "messages": messages,
            **kwargs
        }
    
    def from_output(self, output: t.Dict, inp: Msg | BaseDialog | str | None = None) -> Resp:
        """Convert Chat Completions response to Dachi Resp."""
        choice = output.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        msg = Msg(
            role=message.get("role", "assistant"),
            text=message.get("content", "")
        )
        
        resp = Resp(
            msg=msg,
            text=message.get("content"),
            finish_reason=choice.get("finish_reason"),
            response_id=output.get("id"),
            model=output.get("model"),
            usage=output.get("usage", {}),
            tool=message.get("tool_calls", []) if message.get("tool_calls") else None,
            logprobs=choice.get("logprobs"),
            choices=[{
                "index": c.get("index", i),
                "finish_reason": c.get("finish_reason"),
                "logprobs": c.get("logprobs")
            } for i, c in enumerate(output.get("choices", []))]
        )
        
        # Store provider-specific fields in meta
        resp.meta.update({
            "object": output.get("object"),
            "created": output.get("created"),
            "system_fingerprint": output.get("system_fingerprint"),
            "service_tier": output.get("service_tier"),
        })
        
        # Store any additional fields not explicitly handled
        resp.meta.update({
            k: v for k, v in output.items() 
            if k not in {"choices", "usage", "model", "id", "object", "created", "system_fingerprint"}
        })
        
        # Store raw response
        resp._data = output
        return resp
    
    def from_streamed(self, output: t.Dict, inp: Msg | BaseDialog | str | None = None, prev_resp: Resp | None = None) -> Resp:
        """Handle Chat Completions streaming responses with proper accumulation."""
        choice = output.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        
        # Accumulate text content
        delta_text = delta.get("content", "") or ""
        if prev_resp and prev_resp.msg:
            accumulated_text = prev_resp.msg.text + delta_text
        else:
            accumulated_text = delta_text
        
        # Create new message with accumulated content
        msg = Msg(
            role=delta.get("role", "assistant"),
            text=accumulated_text  # Full accumulated text
        )
        
        # Create delta object for streaming
        resp_delta = RespDelta(
            text=delta_text,  # Just the delta part
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason"),
            usage=output.get("usage")
        )
        
        if prev_resp is None:
            resp = Resp(msg=msg, delta=resp_delta)
        else:
            resp = prev_resp.spawn(msg=msg, data=output)
            resp.delta = resp_delta
        
        return resp
    
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
    
    def forward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        api_input = self.to_input(inp, **kwargs)
        return self.from_output(
            self.client.chat.completions.create(**api_input),
            inp
        )

    async def aforward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        api_input = self.to_input(inp, **kwargs)
        return self.from_output(
            await self.async_client.chat.completions.create(**api_input),
            inp
        )   

    async def astream(self, inp: Msg | BaseDialog, **kwargs) -> t.AsyncIterator[Resp]:
        api_input = self.to_input(inp, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        async for chunk in await self.async_client.chat.completions.create(**api_input):
            resp = self.from_streamed(chunk, inp, prev_resp)
            prev_resp = resp
            yield resp

    def stream(self, inp: Msg | BaseDialog, **kwargs) -> t.Iterator[Resp]:
        api_input = self.to_input(inp, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        for chunk in self.client.chat.completions.create(**api_input):
            resp = self.from_streamed(chunk, inp, prev_resp)
            prev_resp = resp
            yield resp


class OpenAIResp(LLM, AIAdapt):
    """
    Adapter for OpenAI Responses API.
    
    Converts between Dachi's unified message format and OpenAI Responses API format.
    Specialized for reasoning models that provide thinking/reasoning content.
    
    Key Differences from Chat Completions:
    - Handles 'reasoning' field for model thinking process
    - Accumulates both text and thinking content separately during streaming
    - Uses same streaming pattern as OpenAIChat but with dual content streams
    
    Streaming Pattern:
    1. Accumulates text in resp.msg.text, thinking in resp.thinking  
    2. Sets resp.delta.text and resp.delta.thinking to chunk content only
    3. Uses resp.spawn() to maintain both content streams across chunks
    4. Processors use resp.out_store for stateful accumulation
    """
    
    url: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # Create the OpenAI clients
        client_kwargs = {}
        if self.url:
            client_kwargs['base_url'] = self.url
        
        # Only create clients if we have an API key or explicit test mode
        try:
            self.client = openai.Client(**client_kwargs)
            self.async_client = openai.AsyncClient(**client_kwargs)
        except openai.OpenAIError:
            # For testing purposes, set clients to None
            self.client = None
            self.async_client = None

    def to_input(self, inp: Msg | BaseDialog | str, **kwargs) -> t.Dict:
        """Convert Dachi format to Responses API format."""
        # Handle single user message case
        if isinstance(inp, Msg) and inp.role == "user" and "instructions" not in kwargs:
            return {
                "input": inp.text or "",
                **kwargs
            }
        
        # Handle multiple messages (same as Chat Completions)
        if isinstance(inp, str):
            inp = Msg(role="user", text=inp)
        
        if isinstance(inp, Msg):
            messages = [self._convert_message(inp)]
        else:
            messages = [self._convert_message(msg) for msg in inp]
        
        # Add tool messages from ToolUse objects
        for msg in (inp if isinstance(inp, BaseDialog) else [inp]):
            for tool_out in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "content": str(tool_out.result),
                    "tool_call_id": tool_out.tool_call_id
                })
        
        return {
            "messages": messages,
            **kwargs
        }   
     
    def from_output(self, output: t.Dict, inp: Msg | BaseDialog | str | None = None) -> Resp:
        """Convert Responses API response to Dachi Resp."""
        choice = output.get("choices", [{}])[0]
        message = choice.get("message", {})

        resp_id = output.get("id", None)
        
        if isinstance(inp, Msg):
            prev_id = inp.id
        else:
            prev_id = None

        msg = Msg(
            role=message.get("role", "assistant"),
            text=message.get("content", ""),
            id=resp_id,
            prev_id=prev_id
        )
        
        resp = Resp(
            msg=msg,
            text=message.get("content"),
            thinking=output.get("reasoning"),  # Responses API specific
            finish_reason=choice.get("finish_reason"),
            response_id=output.get("id"),
            model=output.get("model"),
            usage=output.get("usage", {}),
            tool=message.get("tool_calls", []) if message.get("tool_calls") else None,
            logprobs=choice.get("logprobs"),
            choices=[{
                "index": c.get("index", i),
                "finish_reason": c.get("finish_reason"),
                "logprobs": c.get("logprobs")
            } for i, c in enumerate(output.get("choices", []))]
        )
        
        # Store provider-specific fields in meta
        resp.meta.update({
            "object": output.get("object"),
            "created": output.get("created"),
            "system_fingerprint": output.get("system_fingerprint"),
            "service_tier": output.get("service_tier"),
        })
        
        # Store any additional fields not explicitly handled
        resp.meta.update({
            k: v for k, v in output.items() 
            if k not in {"choices", "usage", "model", "id", "reasoning", "object", "created", "system_fingerprint"}
        })
        
        # Store raw response
        resp._data = output
        return resp
    
    def from_streamed(self, output: t.Dict, inp: Msg | BaseDialog | str | None = None, prev_resp: Resp | None = None) -> Resp:
        """Handle Responses API streaming responses with proper accumulation."""
        choice = output.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if isinstance(inp, Msg):
            prev_id = inp.id
        else:
            prev_id = None

        # Accumulate text content
        delta_text = delta.get("content", "") or ""
        if prev_resp and prev_resp.msg:
            accumulated_text = prev_resp.msg.text + delta_text
        else:
            accumulated_text = delta_text

        # Accumulate thinking content
        delta_thinking = delta.get("reasoning", "") or ""
        if prev_resp:
            accumulated_thinking = (prev_resp.thinking or "") + delta_thinking
        else:
            accumulated_thinking = delta_thinking

        msg = Msg(
            role=delta.get("role", "assistant"),
            text=accumulated_text,  # Full accumulated text
            id=output.get("id", None),
            prev_id=prev_id
        )
        
        # Create delta object for streaming
        resp_delta = RespDelta(
            text=delta_text,  # Just the delta part
            thinking=delta_thinking,  # Just the delta part
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason"),
            usage=output.get("usage")
        )
        
        if prev_resp is None:
            resp = Resp(msg=msg, delta=resp_delta, thinking=accumulated_thinking)
        else:
            resp = prev_resp.spawn(msg=msg, data=output)
            resp.delta = resp_delta
            resp.thinking = accumulated_thinking  # Full accumulated thinking
        
        return resp
    
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
    
    def forward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        api_input = self.to_input(inp, **kwargs)
        return self.from_output(
            self.client.responses.create(**api_input),
            inp
        )
    
    async def aforward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        api_input = self.to_input(inp, **kwargs)
        return self.from_output(
            await self.async_client.responses.create(**api_input),
            inp
        )
    
    def stream(self, inp, **kwargs) -> t.Iterator[Resp]:
        api_input = self.to_input(inp, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        for chunk in self.client.responses.create(**api_input):
            resp = self.from_streamed(chunk, inp, prev_resp)
            prev_resp = resp
            yield resp

    async def astream(self, inp, **kwargs) -> t.AsyncIterator[Resp]:
        api_input = self.to_input(inp, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        async for chunk in await self.async_client.responses.create(**api_input):
            resp = self.from_streamed(chunk, inp, prev_resp)
            prev_resp = resp
            yield resp