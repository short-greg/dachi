# 1st party 
from abc import ABC, abstractmethod
import typing

# local
from ..msg import Msg, StreamMsg
from ..proc import Module, AsyncModule
from .. import utils


class MR(object):
    """Retrieves from the message (not the meta in the message)"""
    def __init__(self, name: str):
        """Use to retrieve from the base message dict.

        Args:
            name (str): The name of the message key to retrieve
        """
        self.name = name

    def __call__(self, msg):
        return msg[self.name]


class FromMsg(object):
    """Use to get a value from a message. 
    """

    def __init__(self, key: str | MR | typing.List[str | MR | None]):
        """Use to retrieve values from a message

        MR will return from the base of the message, otherwise
        the meta storage is used. None will return the message itself.

        Args:
            key (str | MR | typing.List[str  |  MR]): 
        """
        if isinstance(key, FromMsg):
            key = key.key
        self.key = key

    def __call__(self, msg: Msg, override=None) -> typing.Any:
        """Use to get a value from the message
        The default is to get from the meta dict in the message.

        Args:
            msg (Msg): The message to get a value from
            override (_type_, optional): Whether to override the key to retrieve. Defaults to None.

        Returns:
            typing.Any: The value retrieved form the message
        """
        if isinstance(override, FromMsg):
            override = override.key
        key = override or self.key
        if isinstance(key, str):
            return msg.m[key]
        elif key is None:
            return msg
        elif isinstance(key, MR):
            return key(msg)
        
        return tuple(
            k(msg) if isinstance(k, MR) 
            else msg if k is None
            else msg.m[k]
            for k in key
        )


class ToMsg(Module, AsyncModule, ABC):
    """Converts the input to a message
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        pass

    async def aforward(self, *args, **kwargs) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        return self.forward(*args, **kwargs)


class NullToMsg(Module, AsyncModule, ABC):
    """Converts a message to a message (so actually does nothing)
    """

    def forward(self, msg: Msg) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        return msg

    async def aforward(self, msg: Msg) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        return msg


# class FromMsg(Module, AsyncModule, ABC):
#     """Converts the message to an output
#     """
#     def __init__(self, from_: typing.Union[str, typing.List[str]]):

#         if isinstance(from_, str):
#             from_ = [from_]
#             self._single = True
#         else:
#             self._single = False
#         self._from = from_

#     @abstractmethod
#     def delta(self, resp, delta_store: typing.Dict, 
#               is_streamed: bool=False, is_last: bool=True) -> typing.Any:
#         """Read in the output

#         Args:
#             message (str): The message to read

#         Returns:
#             typing.Any: The output of the reader
#         """
#         pass

#     def forward(self, msg: Msg, delta_store: bool=None) -> typing.Any:
#         delta_store = delta_store or {}
#         resp = [msg['meta'][r] for r in self._from if r]
#         is_undefined = all(r is utils.UNDEFINED for r in resp)
        
#         if self._single:
#             resp = resp[0]
        
#         if is_undefined:
#             return utils.UNDEFINED
#         msg['meta'][self.name] = self.delta(
#             resp, delta_store, msg.is_streamed, msg.is_last
#         )
#         return msg

#     async def aforward(self, msg: Msg, delta_store: bool=None) -> typing.Any:
#         return self.forward(msg, delta_store)


class ToText(ToMsg):
    """Converts the input to a text message
    """

    def __init__(self, role: str='system', field: str='content'):
        """Converts an input to a text message

        Args:
            role (str): The role for the message
            field (str, optional): The name of the field for the text. Defaults to 'content'.
        """
        self.role = role
        self.field = field

    def forward(self, text: str) -> Msg:
        """Create a text message

        Args:
            text (str): The text for the message

        Returns:
            Msg: Converts to a text message
        """
        return Msg(
            role=self.role, 
            **{self.field: text}
        )


class MsgProc(Module, AsyncModule, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """
    last_val = ''

    def __init__(
        self, name: str, from_: typing.Union[str, typing.List[str]]
    ):
        """A module used to process a message

        Args:
            name (str): The name of the output
            from_ (typing.Union[str, typing.List[str]]): 
        """
        self.name = name
        if isinstance(from_, str):
            from_ = [from_]
            self._single = True
        else:
            self._single = False
        self._from = from_

    def post(self, msg: Msg, result, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False):
        """This method is executed after completion.
        The default is to do nothing
        Args:
            msg (Msg): The message
            result: The result of teh post
            streamed (bool, optional): Whether streamed or not. Defaults to False.
            is_last (bool, optional): Whether it is the last element if streamed. Defaults to False.
        """
        pass

    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict, is_streamed: bool=False, is_last: bool=True) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    def forward(self, msg: Msg, delta_store: typing.Dict=None) -> Msg:
        """Processes the message

        Args:
            msg (Msg): The message to process
            delta_store (typing.Dict, optional): The delta store. Defaults to None.

        Returns:
            Msg: The processed message
        """
        delta_store = delta_store if delta_store is not None else {}

        if isinstance(msg, StreamMsg):
            streamed = True
            is_last = msg.is_last
        else:
            streamed = False
            is_last = True

        resp = [msg['meta'][r] for r in self._from]
        is_undefined = all(r is utils.UNDEFINED for r in resp)
        
        if self._single:
            resp = resp[0]
        
        if is_undefined:
            msg.m[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        msg['meta'][self.name] = res = self.delta(
            resp, delta_store, streamed, is_last
        )
        self.post(msg, res, streamed, is_last)
        return msg

    async def aforward(self, msg: Msg) -> Msg:
        """Processes the message asynchronously

        Args:
            msg (Msg): The message to process
            delta_store (typing.Dict, optional): The delta store. Defaults to None.

        Returns:
            Msg: The processed message
        """
        return self.forward(msg)


class MsgProcSeq(Module, AsyncModule):
    """A sequence of message converters
    """

    def __init__(self, procs: typing.List[MsgProc]):
        """Sequence of message converters

        Args:
            procs (typing.List[MsgProc]): The processes to use in processing the message
        """
        super().__init__()
        self.procs = procs

    def forward(self, msg: Msg) -> Msg:
        """Process the message on the sequence

        Args:
            msg (Msg): The message to process

        Returns:
            Msg: The processed message
        """
        for proc in self.procs:
            msg = proc.forward(msg)
        return msg
    
    async def forward(self, msg: Msg) -> Msg:
        """Process the message on the sequence asynchronously

        Args:
            msg (Msg): The message to process

        Returns:
            Msg: The processed message
        """
        for proc in self.procs:
            msg = await proc.aforward(msg)
        return msg
    
    def __getitem__(self, key: str | int) -> MsgProc:
        """Get a message converter by name

        Args:
            key (str): The name of the message converter

        Returns:
            MsgConv: The message converter
        """
        if isinstance(key, int):
            return self.procs[key]
        
        proc = None
        for proc in self.procs:
            if proc.name == key:
                return proc
        raise KeyError(f"Key {key} not found in {self.procs}")
