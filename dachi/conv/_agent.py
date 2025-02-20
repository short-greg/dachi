from .. import adapt as adapt
from ._chat import Chat
import typing
from ..adapt._ai import ToolCall
from ..data._messages import Msg


class ChatAgent(Chat):
    """A component 

    Similar to Chat but has tools that can be executed
    and will execute them automatically.
    # 1) Make it so that it automatically executes tools
    # 2) Register all of the tools
    """

    def __getitem__(self, idx: int) -> Msg:
        """Get a message from the dialog"""
        return self.dialog[idx]
    
    def __setitem__(self, idx: int, message: Msg) -> Msg:
        """Set a message in the dialog"""
        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over the dialog"""
        for m in self.dialog:
            yield m

    def forward(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.Tuple[typing.Union[typing.Any, Msg], typing.Any]:
        """Execute a turn of the chat"""
        done = False
        while not done:
            res, msg = super().forward(
                *args, out_msg=True, **kwargs
            )
            if isinstance(res, ToolCall):
                result = self.tool_set(res)
                self.dialog = self.dialog.insert(res.msg(result))
            else:
                done = True

        if get_msg:
            return res, get_msg
        return res
    
    async def aforward(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.Union[typing.Tuple[typing.Any, Msg], Msg]:
        """Execute a turn of the chat asynchronously"""

        done = False
        while not done:
            res, msg = await super().aforward(
                *args, get_msg=True, **kwargs
            )
            if msg.type_ != 'tool':
                result = self.tool_set(res)
                self.dialog = self.dialog.insert(res.msg(result))
            else:
                done = True

        if get_msg:
            return res, get_msg
        return res

    def stream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.Iterator[typing.Union[typing.Tuple[typing.Any, Msg], Msg]]:
        """Stream a turn of the chat"""

        done = False
        while not done:
            for res, msg in super().stream(*args, get_msg=True, **kwargs):
                if get_msg:
                    yield res, msg
                else:
                    yield res

            if isinstance(res, ToolCall):
                result = self.tool_set(res)
                self.dialog = self.dialog.insert(res.msg(result))
            else:
                done = True

    async def astream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.AsyncIterator[typing.Union[typing.Tuple[typing.Any, Msg], Msg]]:
        """Stream a turn of the chat asynchronously"""
        done = False
        if not done:

            async for res, msg in await super().stream(*args, out_msg=True, **kwargs):
                if msg.type_ == 'data':
                    if get_msg:
                        yield res, get_msg
                    yield res
            if isinstance(res, ToolCall):
                result = self.tool_set(res)
                self.dialog = self.dialog.insert(res.msg(result))
            else:
                done = True


# how to get a verbose output from the agent?
# 1) don't use recursion: loop over the chat streams until it is done.. 
# 2) output the name of the function, yield DELTA, MSG, NAME
#    Then if name is None.. 
# for forward
# return MSG, STEPS
# 3) how to handle multiple function calls? Loop over each function call


# agent.register([f1, f2]) 
# to_openai_tool([f1, f2]) 
# agent could inherit from chat

# 1) Test
# 2) Implement sample openai agent

# agent = LLMAgent(chat=Chat, tool=openai)
# agent.register([f1, f2])
# tool_prep(tool)...  # chat needs to have the tool_prep
# for ... in agent.stream(content=...):
#    ...
# .. # allow it to only include the assistant messages.. or have otehr filters 
