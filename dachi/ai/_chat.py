import typing

from .. import _core as core


class Chat(core.Module):

    def __init__(
        self, dialog: core.Dialog=None, cue_args: typing.Dict=None, mod: core.Module=None
    ):
        super().__init__()
        self.dialog = dialog or core.ListDialog()
        self.cue_args = cue_args or {}
        self.mod = mod

    def cue(self, **kwargs) -> 'Chat':

        return Chat(
            self.model, self.dialog.clone(), 
            {**kwargs}
        )

    def __getitem__(self, idx: int) -> core.Message:

        return self.dialog[idx]
    
    def __setitem__(self, idx: int, message: core.Message) -> core.Message:

        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[core.Message]:

        for m in self.dialog:
            return m

    def forward(self, *args, f: typing.Callable=None, **kwargs) -> typing.Tuple[core.Msg, 'Chat']:

        f_ = f if f is not None else self.mod.forward
        message =  f_(*args, **kwargs)
        dialog = self.dialog.add(message)
        return Chat(
            dialog, {**self.cue_args}, self.mod 
        )
    
    async def aforward(self, *args, f: typing.Callable=None, **kwargs) -> typing.Tuple[core.Msg, 'Chat']:

        f_ = f if f is not None else self.mod.aforward
        message = await f_(*args, **kwargs)
        dialog = self.dialog.add(message)
        return Chat(
            dialog, {**self.cue_args}, self.mod 
        )

    def stream(self, *args, f: typing.Callable=None, **kwargs) -> typing.Iterator[typing.Tuple[core.Msg, 'Chat']]:

        f_ = f if f is not None else self.mod.stream
        for message in f_(*args, **kwargs):
            dialog = self.dialog.add(message)
            yield message.delta, Chat(
                dialog, {**self.cue_args}, self.mod 
            )

    async def astream(self, *args, f: typing.Callable=None, **kwargs) -> typing.AsyncIterator[typing.Tuple[core.Msg, 'Chat']]:

        f_ = f if f is not None else self.mod.astream
        async for message in await f_(*args, **kwargs):
            dialog = self.dialog.add(message)
            yield message.delta, Chat(
                dialog, {**self.cue_args}, self.mod 
            )
