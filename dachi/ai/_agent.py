from .. import _core as core
from ._ai import LLM
from ._chat import Chat
from ._tool import ToolSet, Tool
import typing


class LLMAgent(core.Module):
    """A component 

    Similar to Chat but has tools that can be executed
    and will execute them automatically.
    # 1) Make it so that it automatically executes tools
    # 2) Register all of the tools
    """

    def __init__(
        self,
        func_msg: typing.Callable[[typing.Any], core.Msg], 
        chat: Chat=None, llm: LLM=None,
        pre: core.Module=None,
        post: core.Module=None,
        tools: ToolSet=None
    ):
        """Create an LLMAgent component

        Args:
            dialog (core.Dialog, optional): The dialog to update after each turn. Defaults to None.
            llm (LLM, optional): The llm to use. Defaults to None.
            pre (core.Module, optional): The pre-processing module. Defaults to None.
            post (core.Module, optional): The post-processing module. Defaults to None.
            tools (typing.List[core.Tool], optional): The tools to register. Defaults to None.
        """
        super().__init__()
        self.func_msg = func_msg
        self.chat = chat 
        self.llm = llm
        self.pre = pre or core.MsgConv()
        self.post = post
        self.tools = tools or ToolSet()
        self.register_tools()

    def register_tools(self):
        """Register all tools"""
        for tool in self.tools:
            self.register(tool)

    def register(self, tool: core.Tool):
        """Register a single tool"""
        # Implementation to register the tool
        self.tools.add(tool)

    def __getitem__(self, idx: int) -> core.Message:
        """Get a message from the dialog"""
        return self.dialog[idx]
    
    def __setitem__(self, idx: int, message: core.Message) -> core.Message:
        """Set a message in the dialog"""
        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[core.Message]:
        """Iterate over the dialog"""
        for m in self.dialog:
            yield m

    def forward(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.Tuple[core.Msg, 'LLMAgent']:
        """Execute a turn of the chat"""
        res, msg = self.chat(*args, f, out_msg=True, **kwargs)
        if msg.type_ == 'data':
            if out_msg:
                return res, out_msg
            return res
        out = self.tools(msg.name, msg.arguments)
        response = msg.respond(out)
        return self.forward(**response, out_msg=out_msg, f=f)
    
    async def aforward(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.Tuple[core.Msg, 'LLMAgent']:
        """Execute a turn of the chat asynchronously"""
        res, msg = await self.chat.aforward(*args, f, out_msg=True, **kwargs)
        if msg.type_ == 'data':
            if out_msg:
                return res, out_msg
            return res
        out = await self.tools.aforward(msg.name, msg.arguments)
        response = msg.respond(out)
        return await self.aforward(**response, out_msg=out_msg, f=f)

    def stream(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.Iterator[typing.Tuple[core.Msg, 'LLMAgent']]:
        """Stream a turn of the chat"""
        
        for res, msg in self.chat.stream(*args, f, out_msg=True, **kwargs):
            if msg.type_ == 'data':
                if out_msg:
                    yield res, out_msg
                yield res
        if msg.type_ == 'data':
            return
        out = self.tools(msg.name, msg.arguments)
        response = msg.respond(out)
        for res in self.stream(**response, out_msg=out_msg, f=f):
            yield res

    async def astream(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.AsyncIterator[typing.Tuple[core.Msg, 'LLMAgent']]:
        """Stream a turn of the chat asynchronously"""
        for res, msg in self.chat.stream(*args, f, out_msg=True, **kwargs):
            if msg.type_ == 'data':
                if out_msg:
                    yield res, out_msg
                yield res
        if msg.type_ == 'data':
            return
        out = self.tools(msg.name, msg.arguments)
        response = msg.respond(out)
        
        async for res in await self.astream(**response, out_msg=out_msg, f=f):
            yield res
