import typing

from .. import _core as core
from ._ai import LLM


class Chat(core.Module):

    def __init__(
        self, dialog: core.Dialog=None, llm: LLM=None,
        pre: core.Module=None,
        post: core.Module=None,
    ):
        super().__init__()
        self.dialog = dialog or core.ListDialog()
        self.llm = llm
        self.pre = pre or MsgConv()
        self.post = post

    def __getitem__(self, idx: int) -> core.Message:

        return self.dialog[idx]
    
    def __setitem__(self, idx: int, message: core.Message) -> core.Message:

        self.dialog[idx] = message
        return message

    def __iter__(self) -> typing.Iterator[core.Message]:

        for m in self.dialog:
            return m

    def forward(self, *args, **kwargs) -> typing.Tuple[core.Msg, 'Chat']:

        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        out_msg = self.llm(dialog)
        dialog = self.dialog.append(out_msg)
        out = self.post(out_msg)
        return out, Chat(
            dialog, self.mod,
            self.pre, self.post
        )
    
    async def aforward(self, *args, **kwargs) -> typing.Tuple[core.Msg, 'Chat']:

        in_msg = await self.pre.aforward(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        out_msg = await self.llm.aforward(dialog)
        dialog = self.dialog.append(out_msg)
        out = await self.post.aforward(out_msg)
        return out, Chat(
            dialog, self.mod,
            self.pre, self.post
        )

    def stream(self, *args, **kwargs) -> typing.Iterator[typing.Tuple[core.Msg, 'Chat']]:

        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        for msg in self.llm.stream(dialog):
            dialog = self.dialog.append(msg)
            for d in self.post.stream(msg):
                yield d, Chat(
                    dialog, self.mod,
                    self.pre, self.post
                )

    async def astream(self, *args, **kwargs) -> typing.AsyncIterator[typing.Tuple[core.Msg, 'Chat']]:

        in_msg = self.pre(*args, **kwargs)
        dialog = self.dialog.append(in_msg)
        async for msg in await self.llm.astream(dialog):
            dialog = self.dialog.append(msg)
            async for d in await self.post.astream(msg):
                yield d, Chat(
                    dialog, self.mod,
                    self.pre, self.post
                )


# msg = llm()
