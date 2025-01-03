import typing
from .. import _core as core
from ._ai import LLM


class Chat(core.Module):
    """A component that facilitates chatting
    """

    def __init__(
        self, dialog: core.Dialog=None, llm: LLM=None,
        pre: core.Module=None,
        post: core.Module=None,
    ):
        """Create a Chat component

        Args:
            dialog (core.Dialog, optional): The dialog to update after each turn. Defaults to None.
            llm (LLM, optional): The llm to use. Defaults to None.
            pre (core.Module, optional): The pre-processing module. Defaults to None.
            post (core.Module, optional): The post-processing module. Defaults to None.
        """
        super().__init__()
        self.dialog = dialog or core.ListDialog()
        self.llm = llm
        self.pre = pre or MsgConv()
        self.post = post

    def __getitem__(self, idx: int) -> core.Message:
        """Get a message from the dialog"""
        return self.dialog[idx]
    
    def spawn(self) -> 'Chat':
        """Spawn a new chat"""
        return Chat(
            self.dialog.clone(), self.llm,
            self.pre, self.post
        )
    
    def __setitem__(self, idx: int, message: core.Message) -> core.Message:
        """Set a message in the dialog"""
        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[core.Message]:
        """Iterate over the dialog"""
        for m in self.dialog:
            yield m

    def forward(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.Tuple[core.Msg, 'Chat']:
        """Execute a turn of the chat"""
        f = f or self.llm
        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        res_msg = f(dialog)
        dialog = self.dialog.append(res_msg)
        out = self.post(res_msg)
        return out if not out_msg else out, res_msg
    
    async def aforward(self, *args, f=None, out_msg: bool=False, **kwargs) -> core.Msg:
        """Execute a turn of the chat asynchronously"""
        f = f or self.llm
        if isinstance(f, core.Module):
            f = f.aforward
        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        res_msg = await f(dialog)
        dialog = self.dialog.append(res_msg)
        out = self.post(res_msg)
        return out if not out_msg else out, res_msg

    def stream(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.Iterator[core.Msg]:
        """Stream a turn of the chat"""
        f = f or self.llm
        if isinstance(f, core.Module):
            f = f.stream
        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        for msg in f.stream(dialog):
            for d in self.post.stream(msg):
                yield d if not out_msg else d, msg
        dialog = self.dialog.append(msg)

    async def astream(self, *args, f=None, out_msg: bool=False, **kwargs) -> typing.AsyncIterator[typing.Tuple[core.Msg, 'Chat']]:
        """Stream a turn of the chat asynchronously"""
        f = f or self.llm
        if isinstance(f, core.Module):
            f = f.astream
        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        async for msg in await self.llm.astream(dialog):
            async for d in await self.post.astream(msg):
                yield d if not out_msg else d, msg
        dialog = self.dialog.append(msg)
