# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..core import Msg, Resp

RESPONSE = 'resp'


S = typing.TypeVar('S', bound=pydantic.BaseModel)


# 1st party 
from abc import ABC, abstractmethod
import typing

# local
from . import Msg
from ._process import Process
from .. import utils


class RespGet(Process):
    """Retrieves from the message (not the meta in the message)"""

    @abstractmethod
    def forward(self, msg: Msg) -> typing.Any:
        pass


class KeyGet(RespGet):
    """Retrieves from the message (not the meta in the message)"""

    name: str
    meta: bool = False

    def forward(self, msg) -> typing.Any:
        """

        Args:
            msg: Do a direct retrieval from the message

        Returns:
            typing.Any: The value referenced by key
        """
        if self.meta:
            return msg.m[self.name]
        return msg[self.name]


class TupleGet(RespGet):
    """Retrieves from the message (not the meta in the message)"""

    _keys: typing.List[str]

    def __post_init__(self, keys: typing.Iterable):
        """Use to retrieve from the base message dict.

        Args:
            name (str): The name of the message key to retrieve
        """
        self._rets = [to_get(key) for key in self._keys]

    def forward(self, msg):
        
        return tuple(
            ret(msg)
            for ret in self._rets
        )


def to_get(key: str) -> RespGet:
    """Retrieve a value

    Args:
        key (str): The key to get

    Raises:
        ValueError: 

    Returns:
        MsgGet: The getter
    """
    if isinstance(key, KeyGet):
        return key
    elif isinstance(key, str):
        return KeyGet(key, True)
    elif isinstance(key, typing.Iterable):
        return TupleGet(key)
    
    elif isinstance(key, RespGet):
        return key
    
    raise ValueError(f'Could not convert {key} to a MsgRet')


class RespProc(Process, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """
    last_val = ''
    name: str
    from_: str | typing.List[str]

    def __post_init__(self):
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
        self, resp: Resp, result, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False):
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
    def delta(
        self, resp, delta_store: typing.Dict, 
        is_streamed: bool=False, is_last: bool=True
    ) -> typing.Any:
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
        delta_store: typing.Dict=None,
        is_streamed: bool=False,
        is_last: bool=True
    ) -> Msg:
        """Processes the message

        Args:
            msg (Msg): The message to process
            delta_store (typing.Dict, optional): The delta store. Defaults to None.

        Returns:
            Msg: The processed message
        """
        delta_store = delta_store if delta_store is not None else {}

        resp = [resp.data[r] for r in self.from_]
        is_undefined = all(r is utils.UNDEFINED for r in resp)
        
        if self._single:
            resp = resp[0]
        
        if is_undefined:
            resp.data[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        resp.data[self.name] = res = self.delta(
            resp, delta_store, is_streamed, is_last
        )
        self.post(resp, res, is_streamed, is_last)
        return resp

    @classmethod
    def run(
        cls, 
        resp: Resp, 
        proc: typing.Union['RespProc', None, typing.List['RespProc']], 
        delta_store: typing.List=None
    ) -> 'Msg':
        if proc is None:
            return resp
        if isinstance(proc, RespProc):
            if delta_store is None:
                delta_store = [{}]
            return proc(resp, delta_store=delta_store[0])
        
        if delta_store is None:
            delta_store = [{} for _ in range(len(proc))]
        for p, delta_store_i in zip(proc, delta_store):
            resp = p(resp, delta_store=delta_store_i)
        return resp

    def __call__(
        self, 
        resp: Resp, 
        delta_store: typing.Dict=None, 
        is_streamed: bool=False, 
        is_last: bool=True
    ):
        return super().__call__(
            resp, delta_store, is_streamed, is_last
        )


class RespConv(RespProc, ABC):
    """Use to process the resoponse from an LLM
    """
    conv_env_tok = lambda : {'content': ''}
    from_: str = 'response'

    @abstractmethod
    def delta(
        self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False
    ) -> Msg: 
        pass

    def forward(
        self, 
        resp: Resp, 
        delta_store: typing.Dict=None, 
        is_streamed: bool=False, 
        is_last: bool=True
    ):

        if delta_store is None:
            delta_store = {}
        resp = resp.data[self.from_[0]]
        
        resp.data[self.name] = res = self.delta(
            resp, delta_store, is_streamed, is_last
        )

        self.post(resp, res, delta_store, is_streamed, is_last)
        return resp

    def prep(self) -> typing.Dict:
        return {}
