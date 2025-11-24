# 1st party
import typing as t
from abc import abstractmethod

# 3rd party
import pydantic

# local
from ..core import (
    Msg, Resp, Module, DeltaResp, Prompt,  BaseDialog, BaseTool
)
from ._process import AsyncProcess
from ._resp import ToOut
from ._msg import ToPrompt

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


class AIAdapt(Module):
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

    Note: Tools should be attached to Prompt messages via prompt.tools,
    not passed as parameters. Adapters extract tools using extract_tools_from_messages().
    """
    def forward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
        raise NotImplementedError

    async def aforward(
        self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs
    ) -> Resp:
        raise NotImplementedError

    def stream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        raise NotImplementedError

    def astream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        raise NotImplementedError

L = t.TypeVar("L", bound=LLM)
T = t.TypeVar("T", bound=ToPrompt)

class Op(
    AsyncProcess, Process, StreamProcess, AsyncStreamProcess, t.Generic[L, T]
):
    """
    A basic operation that applies a function to the input.

    Wraps an LLM to provide a simpler interface for common operations:
    - Optionally converts input to messages using to_prompt
    - Optionally prepends system message
    - Handles tools and output processing
    - Returns processed output values directly

    Note: Op is immutable. Dialog is passed as a parameter to methods, not stored as instance state.
    This allows for functional composition and easier state management.
    """

    llm: L
    to_prompt: T | None = None
    system: str | None = None
    base_out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None
    tools: t.List[BaseTool] = []

    def forward(
        self,
        prompt: Prompt | t.Any,
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
        dialog: BaseDialog | None = None,
        **kwargs
    ) -> t.Tuple[t.Any, BaseDialog | None]:
        """
        Processes the input through the LLM and returns the output with updated dialog.

        Args:
            prompt: The input data to be processed. Should be Prompt if to_prompt is None,
                otherwise can be any type that to_prompt can convert.
            out: The output processor or format. Defaults to None.
            dialog: Optional dialog to maintain conversation state. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            Tuple of (processed_output, updated_dialog). If no dialog was provided, the second
            element will be None.
        """
        out = out or self.base_out
        messages = []
        prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

        # Add tools to the prompt message if provided
        if self.tools:
            prompt_message = prompt_message.clone()
            prompt_message.tools = self.tools
            prompt_message.tool_override = True

        if self.system is not None:
            messages.append(Msg(role='system', text=self.system))
        if dialog is not None:
            messages.extend(list(dialog))
        messages.append(prompt_message)

        resp = self.llm.forward(
            messages,
            out=out,
            **kwargs
        )

        # Update dialog if provided
        updated_dialog = dialog
        if dialog is not None:
            dialog.add(prompt_message)
            dialog.add(resp)
            updated_dialog = dialog

        return resp.out, updated_dialog

    async def aforward(
        self,
        prompt: Prompt | t.Any,
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
        dialog: BaseDialog | None = None,
        **kwargs
    ) -> t.Tuple[t.Any, BaseDialog | None]:
        """
        Asynchronously processes the input through the LLM and returns the output with updated dialog.

        Args:
            prompt: The input data to be processed. Should be Prompt if to_prompt is None,
                otherwise can be any type that to_prompt can convert.
            out: The output processor or format. Defaults to None.
            dialog: Optional dialog to maintain conversation state. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            Tuple of (processed_output, updated_dialog). If no dialog was provided, the second
            element will be None.
        """
        out = out or self.base_out
        messages = []
        prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

        # Add tools to the prompt message if provided
        if self.tools:
            prompt_message = prompt_message.clone()
            prompt_message.tools = self.tools
            prompt_message.tool_override = True

        if self.system is not None:
            messages.append(Msg(role='system', text=self.system))
        if dialog is not None:
            messages.extend(list(dialog))
        messages.append(prompt_message)

        resp = await self.llm.aforward(
            messages,
            out=out,
            **kwargs
        )

        # Update dialog if provided
        updated_dialog = dialog
        if dialog is not None:
            dialog.add(prompt_message)
            dialog.add(resp)
            updated_dialog = dialog

        return resp.out, updated_dialog

    def stream(
        self,
        prompt: Prompt | t.Any,
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
        dialog: BaseDialog | None = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[t.Any, BaseDialog | None]]:
        """
        Streams the input through the LLM, yielding outputs as they are produced.

        Args:
            prompt: The input data to be processed. Should be Prompt if to_prompt is None,
                otherwise can be any type that to_prompt can convert.
            out: The output processor or format. Defaults to None.
            dialog: Optional dialog to maintain conversation state. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Yields:
            Iterator of tuples (processed_output, updated_dialog). Dialog is only updated
            on the final yield.
        """
        out = out or self.base_out
        messages = []
        prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

        # Add tools to the prompt message if provided
        if self.tools:
            prompt_message = prompt_message.clone()
            prompt_message.tools = self.tools
            prompt_message.tool_override = True

        if self.system is not None:
            messages.append(Msg(role='system', text=self.system))
        if dialog is not None:
            messages.extend(list(dialog))
        messages.append(prompt_message)

        final_resp = None
        for resp, _delta_resp in self.llm.stream(
            messages,
            out=out,
            **kwargs
        ):
            final_resp = resp
            yield resp.out, None

        # Update dialog with final response
        updated_dialog = dialog
        if dialog is not None and final_resp is not None:
            dialog.add(prompt_message)
            dialog.add(final_resp)
            updated_dialog = dialog

        # Yield final response with updated dialog
        if final_resp is not None:
            yield final_resp.out, updated_dialog

    async def astream(
        self,
        prompt: Prompt | t.Any,
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
        dialog: BaseDialog | None = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[t.Any, BaseDialog | None]]:
        """
        Asynchronously streams the input through the LLM, yielding outputs as they are produced.

        Args:
            prompt: The input data to be processed. Should be Prompt if to_prompt is None,
                otherwise can be any type that to_prompt can convert.
            out: The output processor or format. Defaults to None.
            dialog: Optional dialog to maintain conversation state. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Yields:
            AsyncIterator of tuples (processed_output, updated_dialog). Dialog is only updated
            on the final yield.
        """
        out = out or self.base_out
        messages = []
        prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

        # Add tools to the prompt message if provided
        if self.tools:
            prompt_message = prompt_message.clone()
            prompt_message.tools = self.tools
            prompt_message.tool_override = True

        if self.system is not None:
            messages.append(Msg(role='system', text=self.system))
        if dialog is not None:
            messages.extend(list(dialog))
        messages.append(prompt_message)

        final_resp = None
        async for resp, _delta_resp in await self.llm.astream(
            messages,
            out=out,
            **kwargs
        ):
            final_resp = resp
            yield resp.out, None

        # Update dialog with final response
        updated_dialog = dialog
        if dialog is not None and final_resp is not None:
            dialog.add(prompt_message)
            dialog.add(final_resp)
            updated_dialog = dialog

        # Yield final response with updated dialog
        if final_resp is not None:
            yield final_resp.out, updated_dialog
