import typing
from .. import adapt as adapt
from ..proc import Module
from ._asst import Assistant
from ._messages import Msg, BaseDialog


CHAT_RES = typing.Union[
    typing.Any, typing.Tuple[typing.Any, Msg]
]


class Chat(Module):
    """A component that facilitates chatting
    """
    def __init__(
        self, assistant: Assistant, 
        dialog: BaseDialog=None
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
        self.dialog = dialog or adapt.ListDialog()
        self.assistant = assistant

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
        self, *args, get_msg: bool=False, **kwargs
    ) -> CHAT_RES:
        """Execute a turn of the chat"""
        msg = self.assistant.user(*args, **kwargs)
        dialog = self.dialog.insert(msg)
        out_msg, res = self.assistant(dialog)
        self.dialog = self.dialog.insert(out_msg)
        return out_msg if not get_msg else res, out_msg
    
    async def aforward(
        self, *args, get_msg: bool=False, **kwargs
    ) -> Msg:
        """Execute a turn of the chat asynchronously"""
        in_msg = self.assistant.user(*args, **kwargs)
        dialog = self.dialog.insert(in_msg)
        out_msg, res = await self.assistant.aforward(dialog)
        dialog = self.dialog.insert(out_msg)
        return res if not get_msg else res, out_msg

    def stream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.Iterator[Msg]:
        """Stream a turn of the chat"""
        in_msg = self.assistant.user(*args, **kwargs)
        self.dialog = self.dialog.insert(in_msg)
        for msg, d in self.assistant.stream(self.dialog):
            yield d if not get_msg else d, msg
        self.dialog = self.dialog.insert(msg)

    async def astream(
        self, *args, get_msg: bool=False, **kwargs
    ) -> typing.AsyncIterator[typing.Tuple[Msg, 'Chat']]:
        """Stream a turn of the chat asynchronously"""
        in_msg = self.user(*args, **kwargs)
        self.dialog = self.dialog.insert(in_msg)
        async for msg, d in await self.assistant.astream(self.dialog):
            yield d if not get_msg else d, msg
        self.dialog = self.dialog.insert(msg)
    
    def append(self, msg: Msg):
        """

        Args:
            msg (core.Msg): 
        """
        self.dialog = self.dialog.insert(msg)
