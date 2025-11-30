# 1st party
import typing as t
from abc import abstractmethod, ABC

# 3rd party
import pydantic

# local
from ..core import (
    Msg, Resp, Module, DeltaResp, Prompt,  BaseDialog, BaseTool
)
from ._process import AsyncProcess
from ._resp import ToOut
from ._msg import TO_PROMPT

# TODO: MOVE OUT OF HERE
S = t.TypeVar('S', bound=pydantic.BaseModel)

from ._process import Process, AsyncProcess, AsyncStreamProcess, StreamProcess
from ..core import BaseDialog

# TODO: Update Assistant / LLM
LLM_PROMPT: t.TypeAlias = t.Union[t.Iterable[Msg], Msg]
S = t.TypeVar('S', bound=pydantic.BaseModel)


class LLMAdapter(Process, AsyncProcess, StreamProcess, AsyncStreamProcess, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    async def aforward(self, *args, **kwargs):
        pass

    @abstractmethod
    def stream(self, *args, **kwargs):
        pass

    @abstractmethod
    async def astream(self, *args, **kwargs):
        pass


LLM_ADAPTER = t.TypeVar("LLM_ADAPTER", bound=LLMAdapter)


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


class LangEngine(Process, AsyncProcess, StreamProcess, AsyncStreamProcess, t.Generic[LLM_ADAPTER]):
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
    llm: LLM_ADAPTER

    @abstractmethod
    def to_input(self, msg: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to LLM-specific input format"""
        pass
    
    def forward(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function with format conversion"""
        api_input = self.to_input(msg, *args, **kwargs)
        result = self.llm.forward(**api_input)
        resp = Resp()
        self.set_core_elements(
            resp=resp,
            result=result,
        )
        self.set_tools_delta(
            resp=resp,
            result=result,
        )
        
        # Process with out parameter
        resp.out = self.set_out(resp, out)
        return resp
    
    async def aforward(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function asynchronously with format conversion"""
        api_input = self.to_input(msg, *args, **kwargs)
        result = await self.llm.aforward(**api_input)
        resp = Resp()
        self.set_core_elements(
            resp=resp,
            result=result,
        )
        self.set_tools_delta(
            resp=resp,
            result=result,
        )
        
        # Process with out parameter
        resp.out = self.set_out(resp, out)
        return resp
        
    def stream(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function with format conversion"""
        api_input = self.to_input(msg, *args, stream=True, **kwargs)
        
        prev_resp = None
        delta_store = {}
        for chunk in self.llm.stream(**api_input):
            resp = Resp()
            delta = DeltaResp()
            self.set_core_delta_elements(
                cur_resp=resp,
                cur_delta=delta,
                prev_resp=prev_resp,
                delta_message=chunk
            )
            self.set_tools_delta(
                cur_resp=resp,
                cur_delta=delta,
                prev_resp=prev_resp,
                delta_message=chunk
            )
            resp.out = self.set_out_delta(resp, delta, out, delta_store, is_last=False)  
            yield resp, delta
    
    async def astream(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function asynchronously with format conversion"""
        api_input = self.to_input(msg, *args, stream=True, **kwargs)
        
        prev_resp = None
        delta_store = {}
        async for chunk in await self.llm.astream(**api_input):
            
            resp = Resp()
            delta = DeltaResp()
            self.set_core_delta_elements(
                cur_resp=resp,
                cur_delta=delta,
                prev_resp=prev_resp,
                delta_message=chunk
            )
            self.set_tools_delta(
                cur_resp=resp,
                cur_delta=delta,
                prev_resp=prev_resp,
                delta_message=chunk
            )
            self.set_out_delta(
                cur_resp=resp,
                cur_delta=delta,
                out=out,
                prev_resp=prev_resp,
                delta_message=chunk,
                delta_store=delta_store,  # You might want to replace this with actual delta store
                is_last=False,  # You might want to replace this with actual is_last flag
            )
            yield resp, delta
        # have to set the last response outside the loop to mark is_last=True
        # think how to do that 
        
    @abstractmethod
    def set_core_elements(self, resp: "Resp", message: t.Dict) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ) -> t.Dict[str, t.Any]:
        pass

    @abstractmethod
    def set_tools(self, resp: "Resp", message: dict) -> dict:
        pass

    @abstractmethod
    def set_tools_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ) -> dict:
        pass

    def set_out(
        self,
        resp: Resp,
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None],
    ) -> dict:
        if out is None:
            resp.out = None
        elif isinstance(out, dict):
            resp.out = {key: processor.forward(resp.text) for key, processor in out.items()}
        elif isinstance(out, tuple):
            resp.out = tuple(processor.forward(resp.text) for processor in out)
        elif isinstance(out, ToOut):
            resp.out = out.forward(resp.text)
        else:
            raise TypeError(f"Unsupported out type: {type(out)}")

    def set_out_delta(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None],
        delta_store: dict,
        is_last: bool,
    ) -> dict:
        if out is None:
            cur_resp.out = None
        elif isinstance(out, dict):
            cur_resp.out = {key: processor.delta(cur_delta.text, delta_store, is_last) for key, processor in out.items()}
        elif isinstance(out, tuple):
            cur_resp.out = tuple(processor.delta(cur_delta.text, delta_store, is_last) for processor in out)
        elif isinstance(out, ToOut):
            cur_resp.out = out.delta(cur_delta.text, delta_store, is_last)
        else:
            raise TypeError(f"Unsupported out type: {type(out)}")

    # @abstractmethod  
    # def from_result(self, result: t.Dict, msg: Msg | BaseDialog) -> Resp:
    #     """Convert LLM response to Dachi Resp"""
    #     pass
        
    # @abstractmethod
    # def from_streamed_result(self, result: t.Dict, msg: Msg | BaseDialog, prev_resp: Resp | None) -> t.Tuple[Resp, DeltaResp]:
    #     """Convert streaming LLM response to Dachi Resp + DeltaResp"""
    #     pass

            # Process with out parameter using accumulated text (simplified approach)
            
            # prev_resp = resp
            # yield resp, delta_resp



# class Op(
#     AsyncProcess, Process, StreamProcess, AsyncStreamProcess, t.Generic[LLM_, TO_PROMPT]
# ):
#     """
#     A basic operation that applies a function to the input.

#     Wraps an LLM to provide a simpler interface for common operations:
#     - Optionally converts input to messages using to_prompt
#     - Optionally prepends system message
#     - Handles tools and output processing
#     - Returns processed output values directly

#     Note: Op is immutable. Dialog is passed as a parameter to methods, not stored as instance state.
#     This allows for functional composition and easier state management.
#     """

#     llm: LLM_
#     to_prompt: TO_PROMPT | None = None
#     system: str | None = None
#     base_out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None
#     tools: t.List[BaseTool] = []

#     def forward(
#         self,
#         prompt: Prompt | t.Any,
#         out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
#         dialog: BaseDialog | None = None,
#         **kwargs
#     ) -> t.Tuple[t.Any, BaseDialog | None]:
#         """
#         Processes the input through the LLM and returns the output with updated dialog.

#         Args:
#             prompt: The input data to be processed. Should be Prompt if to_prompt is None,
#                 otherwise can be any type that to_prompt can convert.
#             out: The output processor or format. Defaults to None.
#             dialog: Optional dialog to maintain conversation state. Defaults to None.
#             **kwargs: Additional keyword arguments to pass to the LLM.

#         Returns:
#             Tuple of (processed_output, updated_dialog). If no dialog was provided, the second
#             element will be None.
#         """
#         out = out or self.base_out
#         messages = []
#         prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

#         # Add tools to the prompt message if provided
#         if self.tools:
#             prompt_message = prompt_message.clone()
#             prompt_message.tools = self.tools
#             prompt_message.tool_override = True

#         if self.system is not None:
#             messages.append(Msg(role='system', text=self.system))
#         if dialog is not None:
#             messages.extend(list(dialog))
#         messages.append(prompt_message)

#         resp = self.llm.forward(
#             messages,
#             out=out,
#             **kwargs
#         )

#         # Update dialog if provided
#         updated_dialog = dialog
#         if dialog is not None:
#             dialog.add(prompt_message)
#             dialog.add(resp)
#             updated_dialog = dialog

#         return resp.out, updated_dialog

#     async def aforward(
#         self,
#         prompt: Prompt | t.Any,
#         out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
#         dialog: BaseDialog | None = None,
#         **kwargs
#     ) -> t.Tuple[t.Any, BaseDialog | None]:
#         """
#         Asynchronously processes the input through the LLM and returns the output with updated dialog.

#         Args:
#             prompt: The input data to be processed. Should be Prompt if to_prompt is None,
#                 otherwise can be any type that to_prompt can convert.
#             out: The output processor or format. Defaults to None.
#             dialog: Optional dialog to maintain conversation state. Defaults to None.
#             **kwargs: Additional keyword arguments to pass to the LLM.

#         Returns:
#             Tuple of (processed_output, updated_dialog). If no dialog was provided, the second
#             element will be None.
#         """
#         out = out or self.base_out
#         messages = []
#         prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

#         # Add tools to the prompt message if provided
#         if self.tools:
#             prompt_message = prompt_message.clone()
#             prompt_message.tools = self.tools
#             prompt_message.tool_override = True

#         if self.system is not None:
#             messages.append(Msg(role='system', text=self.system))
#         if dialog is not None:
#             messages.extend(list(dialog))
#         messages.append(prompt_message)

#         resp = await self.llm.aforward(
#             messages,
#             out=out,
#             **kwargs
#         )

#         # Update dialog if provided
#         updated_dialog = dialog
#         if dialog is not None:
#             dialog.add(prompt_message)
#             dialog.add(resp)
#             updated_dialog = dialog

#         return resp.out, updated_dialog

#     def stream(
#         self,
#         prompt: Prompt | t.Any,
#         out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
#         dialog: BaseDialog | None = None,
#         **kwargs
#     ) -> t.Iterator[t.Tuple[t.Any, BaseDialog | None]]:
#         """
#         Streams the input through the LLM, yielding outputs as they are produced.

#         Args:
#             prompt: The input data to be processed. Should be Prompt if to_prompt is None,
#                 otherwise can be any type that to_prompt can convert.
#             out: The output processor or format. Defaults to None.
#             dialog: Optional dialog to maintain conversation state. Defaults to None.
#             **kwargs: Additional keyword arguments to pass to the LLM.

#         Yields:
#             Iterator of tuples (processed_output, updated_dialog). Dialog is only updated
#             on the final yield.
#         """
#         out = out or self.base_out
#         messages = []
#         prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

#         # Add tools to the prompt message if provided
#         if self.tools:
#             prompt_message = prompt_message.clone()
#             prompt_message.tools = self.tools
#             prompt_message.tool_override = True

#         if self.system is not None:
#             messages.append(Msg(role='system', text=self.system))
#         if dialog is not None:
#             messages.extend(list(dialog))
#         messages.append(prompt_message)

#         final_resp = None
#         for resp, _delta_resp in self.llm.stream(
#             messages,
#             out=out,
#             **kwargs
#         ):
#             final_resp = resp
#             yield resp.out, None

#         # Update dialog with final response
#         updated_dialog = dialog
#         if dialog is not None and final_resp is not None:
#             dialog.add(prompt_message)
#             dialog.add(final_resp)
#             updated_dialog = dialog

#         # Yield final response with updated dialog
#         if final_resp is not None:
#             yield final_resp.out, updated_dialog

#     async def astream(
#         self,
#         prompt: Prompt | t.Any,
#         out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None,
#         dialog: BaseDialog | None = None,
#         **kwargs
#     ) -> t.AsyncIterator[t.Tuple[t.Any, BaseDialog | None]]:
#         """
#         Asynchronously streams the input through the LLM, yielding outputs as they are produced.

#         Args:
#             prompt: The input data to be processed. Should be Prompt if to_prompt is None,
#                 otherwise can be any type that to_prompt can convert.
#             out: The output processor or format. Defaults to None.
#             dialog: Optional dialog to maintain conversation state. Defaults to None.
#             **kwargs: Additional keyword arguments to pass to the LLM.

#         Yields:
#             AsyncIterator of tuples (processed_output, updated_dialog). Dialog is only updated
#             on the final yield.
#         """
#         out = out or self.base_out
#         messages = []
#         prompt_message = prompt if self.to_prompt is None else self.to_prompt.forward(prompt)

#         # Add tools to the prompt message if provided
#         if self.tools:
#             prompt_message = prompt_message.clone()
#             prompt_message.tools = self.tools
#             prompt_message.tool_override = True

#         if self.system is not None:
#             messages.append(Msg(role='system', text=self.system))
#         if dialog is not None:
#             messages.extend(list(dialog))
#         messages.append(prompt_message)

#         final_resp = None
#         async for resp, _delta_resp in await self.llm.astream(
#             messages,
#             out=out,
#             **kwargs
#         ):
#             final_resp = resp
#             yield resp.out, None

#         # Update dialog with final response
#         updated_dialog = dialog
#         if dialog is not None and final_resp is not None:
#             dialog.add(prompt_message)
#             dialog.add(final_resp)
#             updated_dialog = dialog

#         # Yield final response with updated dialog
#         if final_resp is not None:
#             yield final_resp.out, updated_dialog



# class LLM(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
#     """
#     Adapter for Large Language Models (LLMs).

#     Note: Tools should be attached to Prompt messages via prompt.tools,
#     not passed as parameters. Adapters extract tools using extract_tools_from_messages().
#     """
#     def forward(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> Resp:
#         raise NotImplementedError

#     async def aforward(
#         self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs
#     ) -> Resp:
#         raise NotImplementedError

#     def stream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
#         raise NotImplementedError

#     def astream(self, inp: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
#         raise NotImplementedError

# LLM_ = t.TypeVar("L", bound=LLM)



# class AIAdapt(Module):
#     """
#     Use to adapt the message from the standard format
#     to the format required by the LLM
#     """

#     @abstractmethod
#     def to_output(
#         self, 
#         output: t.Dict,
#         inp: Msg | BaseDialog | str | None = None,
#     ) -> Resp:
#         pass

#     @abstractmethod
#     def from_streamed(
#         self, 
#         output: t.Dict,
#         inp: Msg | BaseDialog | str | None = None,
#         prev_resp: Resp | None=None
#     ) -> Resp:
#         pass

#     @abstractmethod
#     def to_input(
#         self, 
#         inp: Msg | BaseDialog, 
#         **kwargs
#     ) -> t.Dict:
#         """Convert the input message to the format required by the LLM.

#         Args:
#             msg (Msg | BaseDialog): The input message to convert.

#         Returns:
#             t.Dict: The converted message in the required format.
#         """
#         pass
