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
from ._resp import RESPONSE_FIELD, ToOut
from .. import utils

# TODO: MOVE OUT OF HERE
S = t.TypeVar('S', bound=pydantic.BaseModel)

from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ..core import BaseDialog

# TODO: Update Assistant / LLM
LLM_PROMPT: t.TypeAlias = t.Union[t.Iterable[Msg], Msg]
S = t.TypeVar('S', bound=pydantic.BaseModel)


def get_resp_output(resp: Resp, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None]):
    """Process out parameter for non-streaming functions using forward() method.
    
    Takes an out parameter and processes it by calling the forward() method on each processor.
    Returns the processed result in the same structure as the input (dict, tuple, or single value).
    
    Args:
        resp (Resp): The response object to process
        out (Union[Dict[str, ToOut], Tuple[ToOut, ...], ToOut, None]): The output processors to apply
            - Dict: Keys map to processor results, e.g. {'text': TextOut(), 'summary': SummaryOut()}
            - Tuple: Returns tuple of processor results, e.g. (text_result, summary_result)  
            - Single ToOut: Returns single processor result
            - None: No processing, returns None
            
    Returns:
        Union[Dict, Tuple, Any, None]: Processed result matching input structure:
            - Dict input -> Dict output with same keys
            - Tuple input -> Tuple output with same length
            - Single input -> Single output value
            - None input -> None output
            
    Raises:
        TypeError: If out parameter is not a supported type (dict, tuple, ToOut, or None)
        
    Example:
        >>> resp = Resp(...)
        >>> result = get_resp_output(resp, {'content': TextOut(), 'tokens': TokenOut()})
        >>> # Returns: {'content': 'processed text', 'tokens': 42}
    """
    if out is None:
        return None
    elif isinstance(out, dict):
        return {key: processor.forward(resp) for key, processor in out.items()}
    elif isinstance(out, tuple):
        return tuple(processor.forward(resp) for processor in out)
    elif isinstance(out, ToOut):
        return out.forward(resp)
    else:
        raise TypeError(f"Unsupported out type: {type(out)}")


def get_delta_resp_output(
    resp: Resp, 
    out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None], 
    is_last: bool
):
    """Process out parameter for streaming functions using delta() method.
    
    Uses resp.out_store for state management automatically. Keys are:
    - Dict: uses the original dictionary keys  
    - Tuple: uses numerical keys "0", "1", etc.
    - Single: uses key "val"
    
    Args:
        resp (Resp): The response object to process (uses resp.out_store for state)
        out (Union[Dict[str, ToOut], Tuple[ToOut, ...], ToOut, None]): The output processors to apply
        is_last (bool): Whether this is the final streaming chunk
        
    Returns:
        Union[Dict, Tuple, Any, None]: Processed result matching input structure
            
    Raises:
        TypeError: If out parameter is not a supported type (dict, tuple, ToOut, or None)
    """
    if out is None:
        return None
    elif isinstance(out, dict):
        result = {}
        for key, processor in out.items():
            if key not in resp.out_store:
                resp.out_store[key] = {}
            value = processor.delta(resp, resp.out_store[key], is_last)
            result[key] = value
        return result
    elif isinstance(out, tuple):
        results = []
        for i, processor in enumerate(out):
            key = str(i)
            if key not in resp.out_store:
                resp.out_store[key] = {}
            result = processor.delta(resp, resp.out_store[key], is_last)
            results.append(result)
        return tuple(results)
    elif isinstance(out, ToOut):
        key = "val"
        if key not in resp.out_store:
            resp.out_store[key] = {}
        return out.delta(resp, resp.out_store[key], is_last)
    else:
        raise TypeError(f"Unsupported out type: {type(out)}")




def llm_forward(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, 
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
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = f(
        *args, **kwargs
    )
    resp = _adapt.to_output(result)

    # Process with out parameter
    resp.out = get_resp_output(resp, out)
    
    return resp


async def llm_aforward(
    f, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, 
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
    if _adapt is None:
        _adapt = DefaultAdapter()

    kwargs.update(_adapt.to_input(*args, **kwargs))
    result = await f(
        *args, **kwargs
    )
    resp = _adapt.to_output(result)

    # Process with out parameter
    resp.out = get_resp_output(resp, out)

    return resp


def llm_stream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, 
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
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    
    for response in f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        resp.data[RESPONSE_FIELD] = response

        # Process with out parameter for streaming
        is_last = response == END_TOK
        resp.out = get_delta_resp_output(resp, out, is_last)
        
        prev_message = resp
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    # Process END_TOK with processors to get final accumulated results
    resp.out = get_delta_resp_output(resp, out, True)

    yield resp


async def llm_astream(
    f: t.Callable, 
    *args, 
    _adapt: t.Union['AIAdapt', None] = None,
    out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, 
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
    prev_message: Msg | None = None
    resp = None
    
    if _adapt is None:
        _adapt = DefaultAdapter()
        
    kwargs.update(_adapt.to_input(*args, **kwargs))
    
    async for response in await f(
        *args, **kwargs
    ):
        resp = _adapt.from_streamed(response, resp)
        resp.data[RESPONSE_FIELD] = response

        # Process with out parameter for streaming
        is_last = response == END_TOK
        resp.out = get_delta_resp_output(resp, out, is_last)

        prev_message = resp
        yield resp
    
    msg = Msg(role=_role)
    if prev_message is not None:
        msg.delta = prev_message.delta
    resp.data[RESPONSE_FIELD] = END_TOK

    # Process END_TOK with processors to get final accumulated results
    resp.out = get_delta_resp_output(resp, out, True)

    yield resp


class AIAdapt(BaseModule):
    """
    Use to adapt the message from the standard format
    to the format required by the LLM
    """

    @abstractmethod
    def to_output(
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
    def forward(self, inp: Msg | BaseDialog, out=None,**kwargs) -> Resp:
        raise NotImplementedError

    async def aforward(
        self, inp: Msg | BaseDialog, out=None,
        tools=None, **kwargs
    ) -> Resp:
        raise NotImplementedError

    def stream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.Iterator[Resp]:
        raise NotImplementedError

    def astream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.AsyncIterator[Resp]:
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
    
    def to_output(self, output: t.Dict) -> Resp:
        """Create a basic Resp object from the output."""
        # Extract text content from common patterns
        text_content = None
        if isinstance(output, dict):
            text_content = output.get('content') or output.get('text') or output.get('message')
        elif isinstance(output, str):
            text_content = output
            
        resp = Resp(msg=Msg(role='assistant', text=text_content))
        resp.data['response'] = output
        return resp
    
    def from_streamed(self, output: t.Dict, prev_resp: Resp | None = None) -> Resp:
        """Handle streaming by creating or updating Resp objects."""
        if prev_resp is None:
            accumulated_text = ''
        else:
            accumulated_text = prev_resp.msg.text or ''
        
        # Get delta content for this chunk
        delta_content = ''
        if isinstance(output, dict) and 'content' in output:
            delta_content = output['content']
            accumulated_text += delta_content
        
        if prev_resp is None:
            resp = Resp(msg=Msg(role='assistant', text=accumulated_text))
        else:
            resp = prev_resp.spawn(msg=Msg(role='assistant', text=accumulated_text))
        
        # Set delta text for this chunk
        resp.delta.text = delta_content
        resp.data['response'] = output
        return resp


class Op(AsyncProcess, Process, StreamProcess, AsyncStreamProcess):
    """
    A basic operation that applies a function to the input.
    """

    llm: LLM
    base_out: t.Any
    tools: t.List[t.Any]

    def forward(self, inp: S, out=None, **kwargs) -> S:
        out = out or self.base_out
        return self.llm.delta(inp, tools=self.tools, op=out, **kwargs).out

    async def aforward(self, inp: S, out=None, **kwargs) -> S:
        out = out or self.base_out
        resp = await self.llm.aforward(inp, tools=self.tools, op=out, **kwargs)
        return resp.out

    def stream(self, inp: S, out=None, **kwargs) -> t.Iterator[S]:
        out = out or self.base_out
        for resp in self.llm.stream(inp, tools=self.tools, op=out, **kwargs):

            yield resp.out

    async def astream(self, inp: S, out=None, **kwargs) -> t.AsyncIterator[S]:
        out = out or self.base_out
        async for resp in await self.llm.astream(inp, tools=self.tools, op=out, **kwargs):
            yield resp.out

