# 1st party
import typing as t
from abc import abstractmethod, ABC

# 3rd party
import pydantic

# local
from ..core import (
    Msg, Resp, DeltaResp, Prompt,  BaseDialog, BaseTool
)
from ._process import AsyncProcess, Process, AsyncStreamProcess, StreamProcess
from ._resp import ToOut

# TODO: MOVE OUT OF HERE

from ..core import BaseDialog

# TODO: Update Assistant / LLM
LLM_PROMPT: t.TypeAlias = t.Union[t.Iterable[Msg], Msg]
BASE_MODEL = t.TypeVar('S', bound=pydantic.BaseModel)


class LLMAdapter(Process, AsyncProcess, StreamProcess, AsyncStreamProcess, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) :
        pass
    
    @abstractmethod
    async def aforward(self, *args, **kwargs) -> t.Dict:
        pass

    @abstractmethod
    def stream(self, *args, **kwargs) -> t.Iterator[t.Dict]:
        pass

    @abstractmethod
    async def astream(self, *args, **kwargs) -> t.AsyncIterator:
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

    def forward(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function with format conversion"""
        api_input = self.to_input(msg, *args, **kwargs)
        result = self.llm.forward(**api_input)
        return self.from_resp(result, out)

    async def aforward(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> Resp:
        """Execute LLM function asynchronously with format conversion"""
        api_input = self.to_input(msg, *args, **kwargs)
        result = await self.llm.aforward(**api_input)
        return self.from_resp(result, out)

    def stream(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.Iterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function with format conversion"""
        api_input = self.to_input(msg, *args, stream=True, **kwargs)
        
        prev_resp = None
        delta_store = {}
        for chunk in self.llm.stream(**api_input):
            prev_resp, delta_resp = self.from_streamed_resp(
                prev_resp=prev_resp,
                chunk=chunk,
                out=out,
                delta_store=delta_store,
                is_last=False,  # You might want to replace this with actual is_last flag
            )
            yield prev_resp, delta_resp      
    
    async def astream(self, msg: Msg | BaseDialog, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None, *args, **kwargs) -> t.AsyncIterator[t.Tuple[Resp, DeltaResp]]:
        """Execute streaming LLM function asynchronously with format conversion"""
        api_input = self.to_input(msg, *args, stream=True, **kwargs)
        
        prev_resp = None
        delta_store = {}
        async for chunk in self.llm.astream(**api_input):
            
            prev_resp, delta_resp = self.from_streamed_resp(
                prev_resp=prev_resp,
                chunk=chunk,
                out=out,
                delta_store=delta_store,
                is_last=False,  # You might want to replace this with actual is_last flag
            )
            yield prev_resp, delta_resp
        # have to set the last response outside the loop to mark is_last=True
        # think how to do that 

    def from_resp(self, message: dict, out: t.Union[t.Tuple[ToOut, ...], t.Dict[str, ToOut], ToOut, None] = None) -> Resp:
        """Convert LLM-specific response to Dachi Resp format"""
        resp = Resp()
        self.set_core_elements(
            resp=resp,
            message=message,
        )
        self.set_tools(
            resp=resp,
            message=message,
        )
        self.set_out(resp, out)
        return resp
    
    @abstractmethod
    def to_api_input(self, messages: list[dict], tools: list[dict], format_override: t.Union[bool, dict, pydantic.BaseModel, None], **kwargs) -> t.Dict:
        """Convert Dachi messages, tools, and format override to LLM-specific input format"""
        pass

    def convert_format_override(self, format_override: t.Union[bool, dict, pydantic.BaseModel, None]) -> t.Union[bool, dict, None]:
        """Convert Dachi format override to LLM-specific format override"""
        pass

    def to_input(self, msg: Msg | BaseDialog, **kwargs) -> t.Dict:
        """Convert Dachi messages to LLM-specific input format"""

        tools = extract_tools_from_messages(msg)
        format_override = extract_format_override_from_messages(msg)
        messages = self.convert_messages(msg)
        tools = self.convert_tools(tools)
        format_override = self.convert_format_override(format_override)
        api_input = self.to_api_input(messages, tools, format_override, **kwargs)

        api_input.update(kwargs)
        return api_input

    def from_streamed_resp(self, prev_resp: Resp, chunk, out, delta_store, is_last=False) -> t.Tuple[Resp, DeltaResp]:
        """Convert DeltaResp to full Resp by applying delta to previous response"""
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
        self.set_out_delta(resp, delta, out, delta_store, is_last=is_last)  
        return resp, delta
    
    def convert_messages(self, msg: Msg | BaseDialog) -> t.List[t.Dict]:
        pass

    def convert_tools(self, tools: t.List[BaseTool]) -> t.List[t.Dict]:
        pass

    def set_core_elements(self, resp: "Resp", message: t.Dict):
        pass

    def set_core_delta_elements(
        self,
        cur_resp: "Resp",
        cur_delta: "DeltaResp",
        prev_resp: "Resp",
        delta_message: dict,  # single Responses streaming event payload
    ):
        pass

    def set_tools(self, resp: "Resp", message: dict):
        pass

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
    ):
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
    ):
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
