from abc import ABC, abstractmethod
import typing

from ..msg import Msg, END_TOK, StreamMsg
from ..proc import Module, AsyncModule

from ..base import Templatable
from .. import utils


class ToMsg(Module, AsyncModule, ABC):
    """Converts the input to a message
    """
    @abstractmethod
    def forward(self, *args, **kwargs) -> Msg:
        pass

    async def aforward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FromMsg(Module, AsyncModule, ABC):
    """Converts the message to an output
    """
    def __init__(self, from_: typing.Union[str, typing.List[str]]):

        if isinstance(from_, str):
            from_ = [from_]
            self._single = True
        else:
            self._single = False
        self._from = from_

    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict, 
              is_streamed: bool=False, is_last: bool=True) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    def forward(self, msg: Msg, delta_store: bool=None) -> typing.Any:
        delta_store = delta_store or {}
        resp = [msg['meta'][r] for r in self._from if r]
        is_undefined = all(r is utils.UNDEFINED for r in resp)
        
        if self._single:
            resp = resp[0]
        
        if is_undefined:
            return utils.UNDEFINED
        msg['meta'][self.name] = self.delta(
            resp, delta_store, msg.is_streamed, msg.is_last
        )
        return msg

    async def aforward(self, msg: Msg, delta_store: bool=None) -> typing.Any:
        return self.forward(msg, delta_store)


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


class MsgConv(Module, AsyncModule, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """
    last_val = ''

    def __init__(self, name: str, from_: typing.Union[str, typing.List[str]]):

        self.name = name
        if isinstance(from_, str):
            from_ = [from_]
            self._single = True
        else:
            self._single = False
        self._from = from_

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
            return utils.UNDEFINED
        msg['meta'][self.name] = self.delta(
            resp, delta_store, streamed, is_last
        )
        return msg

    async def aforward(self, msg: Msg) -> Msg:
        return self.forward(msg)


class MsgConvSeq(Module, AsyncModule):
    """A sequence of message converters
    """

    def __init__(self, convs: typing.List[MsgConv]):
        super().__init__()
        self.convs = convs

    def forward(self, msg: Msg) -> Msg:
        for conv in self.convs:
            msg = conv.forward(msg)
        return msg
    
    async def forward(self, msg: Msg) -> Msg:
        for conv in self.convs:
            msg = conv.aforward(msg)
        return msg
    
    def __getitem__(self, key: str | int) -> MsgConv:
        """Get a message converter by name

        Args:
            key (str): The name of the message converter

        Returns:
            MsgConv: The message converter
        """
        if isinstance(key, int):
            return self.convs[key]
        
        conv = None
        for conv in self.convs:
            if conv.name == key:
                return conv
        raise KeyError(f"Key {key} not found in {self.convs}")
