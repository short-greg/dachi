# TODO: Update this

import typing
from ..proc import (
    Module, AsyncModule, 
    StreamModule, AsyncStreamModule
)
from ._ai import Assistant
from ..msg._messages import Msg, BaseDialog, ListDialog


CHAT_RES = typing.Union[
    typing.Any, typing.Tuple[typing.Any, Msg]
]

from ..proc._msg import ToMsg, FromMsg, NullToMsg


class Chat(Module, AsyncModule, StreamModule, AsyncStreamModule):
    """A component that facilitates chatting
    """
    def __init__(
        self, assistant: Assistant, 
        dialog: BaseDialog=None, 
        out: str | typing.List[str] | FromMsg = None,
        to_msg: ToMsg=None
    ):
        """Create a Chat component

        Args:
            dialog (core.Dialog, optional): The dialog to update 
              after each turn. Defaults to None.
            llm (LLM, optional): The llm to use. Defaults to None.
            pre (core.Module, optional): The pre-processing module. Defaults to None.
            post (core.Module, optional): The post-processing module. Defaults to None.
        """
        super().__init__()
        self.dialog = dialog or ListDialog()
        self.assistant = assistant
        self.out = out or FromMsg(None)
        self.to_msg = to_msg or NullToMsg()

    def __getitem__(self, idx: int) -> Msg:
        """Get a message from the dialog"""
        return self.dialog[idx]
    
    def spawn(self) -> 'Chat':
        """Spawn a new chat"""
        return Chat(
            self.dialog.clone(), self.assistant
        )
    
    def __setitem__(
        self, idx: int, message: Msg
    ) -> Msg:
        """Set a message in the dialog"""
        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over the dialog"""
        for m in self.dialog:
            yield m

    def forward(
        self, *args, **kwargs
    ) -> CHAT_RES:
        """Execute a turn of the chat"""
        in_msg = self.to_msg(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        out_msg = self.assistant(dialog)
        dialog = self.dialog.append(out_msg)
        return self.out(out_msg)
    
    async def aforward(
        self, *args, **kwargs
    ) -> Msg:
        """Execute a turn of the chat asynchronously"""

        in_msg = await self.to_msg.aforward(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        out_msg = await self.assistant.aforward(dialog)
        dialog = self.dialog.append(out_msg)
        return self.out(out_msg)

    def stream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.Iterator[Msg]:
        """Stream a turn of the chat"""
        in_msg = self.to_msg(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        for out_msg in self.assistant.stream(dialog):
            yield self.out(out_msg)
        self.dialog.append(out_msg)

    async def astream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.AsyncIterator[typing.Tuple[Msg, 'Chat']]:
        """Stream a turn of the chat asynchronously"""
        in_msg = self.to_msg(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        async for out_msg in await self.assistant.astream(dialog):
            yield self.out(out_msg)
        self.dialog.append(out_msg)
    
    def append(self, msg: Msg):
        """

        Args:
            msg (core.Msg): 
        """
        self.dialog = self.dialog.append(msg)
