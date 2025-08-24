# 1st party
import typing as t
from abc import abstractmethod

# 3rd party
import pydantic

# local
from ..core import (
    Msg, END_TOK, Resp,
    BaseModule, RespDelta
)
from ._process import AsyncProcess
from ._resp import RespProc, RESPONSE_FIELD

# TODO: MOVE OUT OF HERE
S = t.TypeVar('S', bound=pydantic.BaseModel)

from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ..core import BaseDialog

# TODO: Update Assistant / LLM
LLM_PROMPT: t.TypeAlias = t.Union[t.Iterable[Msg], Msg]
S = t.TypeVar('S', bound=pydantic.BaseModel)


def _prepare(proc, kwargs):

    if isinstance(proc, RespProc):
        proc = [proc]
    elif proc is None:
        proc = []
    for r in proc:
        if isinstance(proc, RespProc):
            kwargs.update(r.prep())
    return proc


def llm_forward(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> Resp:
    """
    Executes a given function with specified arguments and processes the response.
    Args:
        f (t.Callable): The function to be executed.
        *args: Variable length argument list to be passed to the function.
        _resp_proc (t.List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to be passed to the function.
    Returns:
        tuple: A tuple containing the final message (Msg) and the last value processed by the Response objects.
    """
    _proc = _prepare(_proc, kwargs)
    
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = f(
        *args, **kwargs
    )
    resp = _adapt.from_output(result)

    for r in _proc:
        resp = r(resp)
    return resp


async def llm_aforward(
    f, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> Resp:
    """
    Asynchronously forwards a function call with additional response processing.
    Args:
        f (Callable): The function to be called.
        *args: Positional arguments to pass to the function.
        _resp_proc (List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        Tuple[Msg, Any]: A tuple containing the processed message and the final value from the response processing.
    """
    _proc = _prepare(_proc, kwargs)
    
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = await f(
        *args, **kwargs
    )
    resp = _adapt.from_output(result)

    for r in _proc:
        if isinstance(r, AsyncProcess):
            resp = await r.aforward(resp)
        else:
            resp = r(resp)

    return resp


def llm_stream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> t.Iterator[Resp]:
    """
    Streams responses from a language model function, allowing for intermediate processing.
    Args:
        f (t.Callable): The language model function to call.
        *args: Positional arguments to pass to the language model function.
        _resp_proc (t.List[Response], optional): A list of Response objects for processing the model's output. Defaults to None.
        _role (str, optional): The role to assign to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the language model function.
    Yields:
        Tuple[Msg, Any]: A tuple containing the message object and the processed value from the response.
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    for response in f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        # msg = Msg(role=_role)
        # if prev_message is not None:
        #     msg.delta = prev_message.delta

        # if resp is None:
        #     resp = Resp(msg=msg)
        # else:
        #     resp = resp.spawn(
        #         msg=msg
        #     )
        resp.data[RESPONSE_FIELD] = response

        for r in _proc:
            resp = r(resp, True, False)
        prev_message = resp
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp


async def llm_astream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    _proc: t.List[RespProc] | RespProc=None, 
    _role: str='assistant',
    **kwargs
) -> t.AsyncIterator[Resp]:
    """

    Args:
        f (t.Callable): The function to run
        _resp_proc (t.List[Response], optional): The processes to use for responding
        _role (str, optional): The role for message. Defaults to 'assistant'.

    Returns:
        t.AsyncIterator: 

    Yields:
        t.AsyncIterator: The Message and the results
    """
    _proc = _prepare(_proc, kwargs)
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    async for response in await f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        # msg = Msg(role=_role)
        # if prev_message is not None:
        #     msg.delta = prev_message.delta

        # if resp is None:
        #     resp = Resp(msg=msg)
        # else:
        #     resp = resp.spawn(
        #         msg=msg
        #     )

        resp.data[RESPONSE_FIELD] = response

        for r in _proc:
            if isinstance(r, AsyncProcess):
                resp = await r(resp, True, False)
            else:
                resp = r(resp, True, False)

        prev_message = resp.msg
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    for r in _proc:

        resp = r(resp, True, True)

    yield resp


class AIAdapt(BaseModule):
    """
    Use to adapt the message from the standard format
    to the format required by the LLM
    """

    @abstractmethod
    def from_output(
        self, 
        output: t.Dict,
        inp: Msg | BaseDialog | str | None = None,
    ) -> Resp:
        pass

    @abstractmethod
    def from_streamed(
        self, 
        output: t.Dict,
        inp: Msg | BaseDialog | str | None = None,
        prev_resp: Resp | None=None
    ) -> Resp:
        pass

    @abstractmethod
    def to_input(
        self, 
        inp: Msg | BaseDialog, 
        **kwargs
    ) -> t.Dict:
        """Convert the input message to the format required by the LLM.

        Args:
            msg (Msg | BaseDialog): The input message to convert.

        Returns:
            t.Dict: The converted message in the required format.
        """
        pass


class LLM(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """
    Adapter for Large Language Models (LLMs).
    """
    def forward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        raise NotImplementedError

    async def aforward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        raise NotImplementedError

    def stream(self, inp: Msg | BaseDialog, **kwargs) -> t.Iterator[Resp]:
        raise NotImplementedError
    
    def astream(self, inp: Msg | BaseDialog, **kwargs) -> t.AsyncIterator[Resp]:
        raise NotImplementedError


class DefaultAdapter(AIAdapt):
    """
    Default/Null adapter that passes data through without transformation.
    
    Used when no specific adapter is needed, providing minimal processing
    while maintaining the AIAdapt interface.
    """
    
    def to_input(self, inp: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Pass through kwargs without transformation."""
        return kwargs
    
    def from_output(self, output: t.Dict) -> Resp:
        """Create a basic Resp object from the output."""
        resp = Resp(msg=Msg(role='assistant'))
        resp.data['response'] = output
        return resp
    
    def from_streamed(self, output: t.Dict, prev_resp: Resp | None = None) -> Resp:
        """Handle streaming by creating or updating Resp objects."""
        if prev_resp is None:
            resp = Resp(msg=Msg(role='assistant'))
        else:
            resp = prev_resp.spawn(msg=Msg(role='assistant'))
        
        resp.data['response'] = output
        return resp


class OpenAIChat(LLM, AIAdapt):
    """
    Adapter for OpenAI Chat Completions API.
    
    Converts between Dachi's unified message format and OpenAI Chat Completions API format.
    
    Message Conversion:
    - Msg.role -> message.role (user/assistant/system/tool)
    - Msg.text -> message.content 
    - Msg.attachments -> message.content (for vision/multimodal)
    - Msg.tool_outs -> role="tool" messages with tool_call_id
    
    Unified kwargs (converted):
    - temperature, max_tokens, top_p, frequency_penalty, presence_penalty
    - stream, stop, seed, user
    
    API-specific kwargs (passed through):
    - tools, tool_choice, response_format, logprobs, top_logprobs
    - parallel_tool_calls, service_tier, stream_options
    """

    def __post_init__(self):
        super().__post_init__()
        # Create the OpenAI Client

    def to_input(self, inp: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi format to Chat Completions format.
        
        Args:
            inp: Single message or dialog to convert
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict with 'messages' array and processed kwargs
        """
        if isinstance(inp, Msg):
            messages = [self._convert_message(inp)]
        else:
            messages = [self._convert_message(msg) for msg in inp]
        
        # Add tool messages from ToolOut objects
        for msg in (inp if isinstance(inp, BaseDialog) else [inp]):
            for tool_out in msg.tool_outs:
                messages.append({
                    "role": "tool",
                    "content": str(tool_out.result),
                    "tool_call_id": tool_out.tool_call_id
                })
        
        return {
            "messages": messages,
            **kwargs
        }
    
    def from_output(
        self, 
        output: t.Dict,
        inp: Msg | BaseDialog | str | None = None,
    ) -> Resp:
        """Convert Chat Completions response to Dachi Resp.
        
        Args:
            output: Raw OpenAI API response
            
        Returns:
            Resp object with unified fields populated
        """
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
            tool=message.get("tool_calls", []) if message.get("tool_calls") else None
        )
        
        # Store provider-specific fields in meta
        resp.meta.update({
            k: v for k, v in output.items() 
            if k not in {"choices", "usage", "model", "id"}
        })
        
        # Store raw response
        resp._data = output
        return resp
    
    def from_streamed(
        self, 
        output: t.Dict, 
        inp: Msg | BaseDialog | str | None = None,
        prev_resp: Resp | None = None
    ) -> Resp:
        """Handle Chat Completions streaming responses.
        
        Args:
            output: Streaming chunk from OpenAI API
            prev_resp: Previous response for delta accumulation
            
        Returns:
            New Resp object spawned from prev_resp or fresh Resp
        """
        choice = output.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        
        # Create new message with delta content
        msg = Msg(
            role=delta.get("role", "assistant"),
            text=delta.get("content", "") or ""
        )
        
        # Create delta object for streaming
        resp_delta = RespDelta(
            text=delta.get("content"),
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason")
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
                    # Convert to OpenAI vision format
                    image_url = attachment.data
                    if not image_url.startswith("data:"):
                        # Assume base64, add data URL prefix
                        mime = attachment.mime or "image/png"
                        image_url = f"data:{mime};base64,{attachment.data}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            openai_msg["content"] = content
        
        return openai_msg
    
    def forward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        inp = self.to_input(inp, **kwargs)
        # execute the request
        return self.from_output(
            self.client.chat.completions.create(**inp),
            inp
        )

    async def aforward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        inp = self.to_input(inp, **kwargs)
        return self.from_output(
            await self.async_client.chat.completions.create(**inp),
            inp
        )   

    async def astream(self, inp: Msg | BaseDialog, **kwargs) -> t.AsyncIterator[Resp]:
        inp = self.to_input(inp, **kwargs)
        async for chunk in self.async_client.chat.completions.create(**inp):
            yield self.from_streamed(chunk, inp)

    def stream(self, inp: Msg | BaseDialog, **kwargs) -> t.Iterator[Resp]:
        inp = self.to_input(inp, **kwargs)
        for chunk in self.client.chat.completions.create(**inp):
            yield self.from_streamed(chunk, inp)


class OpenAIResp(LLM, AIAdapt):
    """
    Adapter for OpenAI Responses API.
    
    Converts between Dachi's unified message format and OpenAI Responses API format.
    
    Message Conversion:
    - Single Msg with role="user" -> input field
    - Multiple messages -> messages array (same as Chat Completions)
    - instructions kwarg -> instructions field (replaces system messages)
    
    Unified kwargs (converted):
    - temperature, max_tokens, top_p, frequency_penalty, presence_penalty
    - stream, stop, seed, user
    
    API-specific kwargs (passed through):
    - instructions, context, tools, tool_choice, response_format
    - modalities, audio, parallel_tool_calls, metadata
    """

    def to_input(self, inp: Msg | BaseDialog | str, **kwargs) -> t.Dict:
        """Convert Dachi format to Responses API format.
        
        Args:
            inp: Single message or dialog to convert
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict with 'input' or 'messages' and processed kwargs
        """
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
        
        # Add tool messages from ToolOut objects
        for msg in (inp if isinstance(inp, BaseDialog) else [inp]):
            for tool_out in msg.tool_outs:
                messages.append({
                    "role": "tool",
                    "content": str(tool_out.result),
                    "tool_call_id": tool_out.tool_call_id
                })
        
        return {
            "messages": messages,
            **kwargs
        }   
     
    def from_output(
        self, 
        output: t.Dict, 
        inp: Msg | BaseDialog | str | None = None
    ) -> Resp:
        """Convert Responses API response to Dachi Resp.
        
        Args:
            output: Raw OpenAI API response
            
        Returns:
            Resp object with unified fields populated
        """
        # Responses API has similar structure to Chat Completions
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
            tool=message.get("tool_calls", []) if message.get("tool_calls") else None
        )
        
        # Store provider-specific fields in meta
        resp.meta.update({
            k: v for k, v in output.items() 
            if k not in {"choices", "usage", "model", "id", "reasoning"}
        })
        
        # Store raw response
        resp._data = output
        return resp
    
    def from_streamed(
        self, output: t.Dict, 
        inp: Msg | BaseDialog | str | None = None,
        prev_resp: Resp | None = None
    ) -> Resp:
        """Handle Responses API streaming responses.
        
        Args:
            output: Streaming chunk from OpenAI API
            prev_resp: Previous response for delta accumulation
            
        Returns:
            New Resp object spawned from prev_resp or fresh Resp
        """
        # Responses API streaming should be similar to Chat Completions
        choice = output.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if isinstance(inp, Msg):
            prev_id = inp.id
        else:
            prev_id = None

        msg = Msg(
            role=delta.get("role", "assistant"),
            text=delta.get("content", "") or "",
            id=output.get("id", None),
            prev_id=prev_id
        )
        
        # Create delta object for streaming
        resp_delta = RespDelta(
            text=delta.get("content"),
            thinking=delta.get("reasoning"),  # Responses API may have reasoning in delta
            tool=delta.get("tool_calls", [{}])[0].get("function", {}).get("arguments") if delta.get("tool_calls") else None,
            finish_reason=choice.get("finish_reason")
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
        inp = self.to_input(inp, **kwargs)
        return self.from_output(
            self.client.responses.create(**inp),
            inp
        )
    
    async def aforward(
        self, 
        inp: Msg | BaseDialog, 
        **kwargs
    ) -> Resp:
        inp = self.to_input(inp, **kwargs)
        return self.from_output(
            await self.async_client.responses.create(**inp),
            inp
        )
    
    def stream(self, inp, **kwargs) -> t.Iterator[Resp]:
        inp = self.to_input(inp, **kwargs)
        for chunk in self.client.responses.create(**inp):
            yield self.from_streamed(chunk, inp)

    async def astream(self, inp, **kwargs) -> t.AsyncIterator[Resp]:
        inp = self.to_input(inp, **kwargs)
        async for chunk in self.async_client.responses.create(**inp):
            yield self.from_streamed(chunk, inp)
