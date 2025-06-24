# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..core import Msg, Resp
from .. import store

RESPONSE = 'resp'
S = typing.TypeVar('S', bound=pydantic.BaseModel)

# 1st party 
from abc import ABC, abstractmethod
import typing

# local
from . import Msg
from ..proc import Module
from .. import utils


class ToMsg(Module, ABC):
    """Converts the input to a message
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        pass


class NullToMsg(ToMsg):
    """Converts a message to a message (so actually does nothing)
    """

    def forward(self, msg: Msg) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        return msg


class ToText(ToMsg):
    """Converts the input to a text message
    """

    role: str = 'system'
    field: str = 'content'

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


class MsgProc(Module, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """
    last_val: typing.ClassVar[str] = ''
    name: str
    from_: str | typing.List[str]

    def __post_init__(
        self
    ):
        """A module used to process a message

        Args:
            name (str): The name of the output
            from_ (typing.Union[str, typing.List[str]]): 
        """
        if isinstance(self.from_, str):
            self.from_ = [self.from_]
            self._single = True
        else:
            self._single = False

    def post(
        self, msg: Msg, result, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False):
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

    def forward(
        self, 
        resp: Resp, 
        is_streamed: bool=False,
        is_last: bool=True
    ) -> Resp:
        """Processes the message

        Args:
            msg (Msg): The message to process
            delta_store (typing.Dict, optional): The delta store. Defaults to None.

        Returns:
            Msg: The processed message
        """
        r = [resp.data[r] for r in self._from]
        is_undefined = all(r is utils.UNDEFINED for r in r)
        
        if self._single:
            r = r[0]
        
        if is_undefined:
            r.data[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        
        delta_store = store.get_or_set(
            r.delta, self.name, {}
        )
        r.data[self.name] = res = self.delta(
            r, delta_store, is_streamed, is_last
        )
        self.post(r, res, is_streamed, is_last)
        return r

    @classmethod
    def run(
        cls, 
        resp: Resp, 
        proc: typing.Union['MsgProc', None, typing.List['MsgProc']], 
        delta_store: typing.List=None
    ) -> 'Msg':
        if proc is None:
            return msg
        if isinstance(proc, MsgProc):
            if delta_store is None:
                delta_store = [{}]
            return proc(resp)
        
        for p in proc:
            msg = p(resp)
        return msg

    def __call__(
        self, 
        resp: Resp, 
        is_streamed: bool=False, 
        is_last: bool=True
    ):
        return super().__call__(
            resp, is_streamed, is_last
        )

