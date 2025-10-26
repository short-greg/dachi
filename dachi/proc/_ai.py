# 1st party
import typing as t
from abc import abstractmethod

# 3rd party
import pydantic

# local
from ..core import (
    Msg, Resp, BaseModule, DeltaResp, Prompt
)
from ._process import AsyncProcess
from ._resp import ToOut

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
        return {key: processor.forward(resp.text) for key, processor in out.items()}
    elif isinstance(out, tuple):
        return tuple(processor.forward(resp.text) for processor in out)
    elif isinstance(out, ToOut):
        return out.forward(resp.text)
    else:
        raise TypeError(f"Unsupported out type: {type(out)}")


# Universal Helper Functions for Message Analysis

def extract_tools_from_messages(messages: Msg | BaseDialog) -> t.List:
    """Universal function that loops through ALL messages to build final tools list
    
    Args:
        messages: Single Msg or BaseDialog (list of messages)
        
    Returns:
        Final accumulated tools list after processing override/extend logic
    """
    tools = []
    
    # Convert single message to list for uniform processing
    msg_list = [messages] if isinstance(messages, Msg) else list(messages)
    
    for msg in msg_list:
        if isinstance(msg, Prompt) and msg.tools is not None:
            if msg.tool_override:
                # Replace entire tools list
                tools = msg.tools.copy()
            else:
                # Extend existing tools list
                tools.extend(msg.tools)
    
    return tools


def extract_format_override_from_messages(messages: Msg | BaseDialog) -> t.Union[bool, dict, pydantic.BaseModel, None]:
    """Universal function that loops through ALL messages to find final format_override
    
    Args:
        messages: Single Msg or BaseDialog (list of messages)
        
    Returns:
        Final format_override (later Prompts override earlier ones)
    """
    format_override = None
    
    # Convert single message to list for uniform processing  
    msg_list = [messages] if isinstance(messages, Msg) else list(messages)
    
    for msg in msg_list:
        if isinstance(msg, Prompt) and msg.format_override is not None:
            # Later Prompts override earlier ones
            format_override = msg.format_override
            
    return format_override


class LLMAdapter(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """
    Base LLM adapter with function injection pattern.
    
    This class provides a unified interface for LLM adapters that handle format conversion
    and execution coordination. Subclasses implement specific format conversion logic.
    
    Key features:
        - Function injection: Pass LLM functions as parameters instead of inheritance
        - Format conversion: Convert between Dachi and LLM-specific formats
        - Streaming support: Handle streaming responses without complex state management
        - Modular design: Separate conversion logic from execution logic
    """
    
    @abstractmethod
    def to_input(self, messages: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to LLM-specific input format"""
        pass
        
    @abstractmethod  
    def from_result(self, result: t.Dict, messages: Msg | BaseDialog) -> Resp:
        """Convert LLM response to Dachi Resp"""
        pass
        
    @abstractmethod
    def from_streamed_result(self, result: t.Dict, messages: Msg | BaseDialog, prev_resp: Resp | None) -> t.Tuple[Resp, DeltaResp]:
        """Convert streaming LLM response to Dachi Resp + DeltaResp"""
        pass
    
    def forward(self, llm_func: t.Callable, messages: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function with format conversion"""
        api_input = self.to_input(messages, **kwargs)
        result = llm_func(**api_input)
        resp = self.from_result(result, messages)
        
        # Process with out parameter
        resp.out = get_resp_output(resp, out)
        return resp
    
    async def aforward(self, llm_func: t.Callable, messages: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function asynchronously with format conversion"""
        api_input = self.to_input(messages, **kwargs)
        result = await llm_func(**api_input)
        resp = self.from_result(result, messages)
        
        # Process with out parameter
        resp.out = get_resp_output(resp, out)
        return resp
        
    def stream(self, llm_func: t.Callable, messages: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function with format conversion"""
        api_input = self.to_input(messages, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        for chunk in llm_func(**api_input):
            resp, delta_resp = self.from_streamed_result(chunk, messages, prev_resp)
            
            # Process with out parameter using accumulated text (simplified approach)
            resp.out = get_resp_output(resp, out)
            
            prev_resp = resp
            yield resp, delta_resp
    
    async def astream(self, llm_func: t.Callable, messages: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function asynchronously with format conversion"""
        api_input = self.to_input(messages, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        async for chunk in await llm_func(**api_input):
            resp, delta_resp = self.from_streamed_result(chunk, messages, prev_resp)
            
            # Process with out parameter using accumulated text (simplified approach)
            resp.out = get_resp_output(resp, out)
            
            prev_resp = resp
            yield resp, delta_resp


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

    def stream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        raise NotImplementedError

    def astream(self, inp: Msg | BaseDialog, out=None, tools=None, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        raise NotImplementedError


class Op(AsyncProcess, Process, StreamProcess, AsyncStreamProcess):
    """
    A basic operation that applies a function to the input.
    """

    llm: LLM
    base_out: t.Any
    tools: t.List[t.Any]

    def forward(self, inp: S, out=None, **kwargs) -> S:
        """
        Processes the input through the LLM and returns the output
        Args:
            inp (S): The input data to be processed.
            out (_type_, optional): The output processor or format. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.
        Returns:
            S: The processed output data.
        """
        out = out or self.base_out
        return self.llm.delta(inp, tools=self.tools, op=out, **kwargs).out

    async def aforward(self, inp: S, out=None, **kwargs) -> S:
        """
        Asynchronously processes the input through the LLM and returns the output
        Args:
            inp (S): _description_
            out (_type_, optional): _description_. Defaults to None.
            **kwargs: _description_
        Returns:
            S: _description_
        """
        out = out or self.base_out
        resp = await self.llm.aforward(inp, tools=self.tools, op=out, **kwargs)
        return resp.out

    def stream(self, inp: S, out=None, **kwargs) -> t.Iterator[S]:
        """
        Streams the input through the LLM, yielding outputs as they are produced.
        Args:
            inp (S): The input data to be processed.
            out (_type_, optional): The output processor or format. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.
        Yields:
            Iterator[S]: An iterator that yields processed outputs.
        """
        out = out or self.base_out
        for resp in self.llm.stream(inp, tools=self.tools, op=out, **kwargs):

            yield resp.out

    async def astream(self, inp: S, out=None, **kwargs) -> t.AsyncIterator[S]:
        """

        Asynchronously streams the input through the LLM, yielding outputs as they are produced.
        Args:
            inp (S): The input data to be processed.
            out (_type_, optional): The output processor or format. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.
        Yields:
            AsyncIterator[S]: An asynchronous iterator that yields processed outputs.
        """
        out = out or self.base_out
        async for resp in await self.llm.astream(inp, tools=self.tools, op=out, **kwargs):
            yield resp.out

