
# 1st party 
from abc import ABC, abstractmethod
import typing

# local
from ..core import Msg, Resp
from ._process import Process
from .. import utils


class RespProc(Process, ABC):
    """Use a reader to read 
    in data convert data retrieved from
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
        self, 
        resp: Resp, 
        result, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
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
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
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

        delta_store = utils.get_or_set(
            resp.delta, self.name, {}
        )
        r = [resp.data[r] for r in self._from]
        is_undefined = all(r is utils.UNDEFINED for r in r)
        
        if self._single:
            r = r[0]
        
        if is_undefined:
            r.data[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        
        # delta_store = utils.get_or_set(
        #     r.delta, self.name, {}
        # )
        r.data[self.name] = res = self.delta(
            r, delta_store, is_streamed, is_last
        )
        self.post(r, res, delta_store, is_streamed, is_last)
        return r

    @classmethod
    def run(
        cls, 
        resp: Resp, 
        proc: typing.Union['RespProc', None, typing.List['RespProc']]
    ) -> 'Resp':
        if proc is None:
            return resp
        if isinstance(proc, RespProc):
            return proc(resp)
        
        for p in proc:
            resp = p(resp)
        return resp

    def __call__(
        self, 
        resp: Resp, 
        is_streamed: bool=False, 
        is_last: bool=True
    ):
        return super().__call__(
            resp, is_streamed, is_last
        )


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
        self, 
        resp: Resp, 
        result, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
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
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
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
        is_undefined = all(
            r is utils.UNDEFINED for r in resp
        )
        
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
    ) -> 'Resp':
        if proc is None:
            return resp
        if isinstance(proc, RespProc):
            if delta_store is None:
                delta_store = [{}]
            return proc(
                resp, delta_store=delta_store[0]
            )
        
        if delta_store is None:
            delta_store = [{} for _ in range(len(proc))]
        for p, delta_store_i in zip(proc, delta_store):
            resp = p(resp, delta_store=delta_store_i)
        return resp

    def __call__(
        self, 
        resp: Resp, 
        is_streamed: bool=False, 
        is_last: bool=True
    ):
        return super().__call__(
            resp, is_streamed, is_last
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
        is_streamed: bool=False, 
        is_last: bool=True
    ):

        r = resp.data[self.from_[0]]

        utils.get_or_set(resp.delta, self.name, {})
        
        resp.data[self.name] = res = self.delta(
            r, 
            resp.delta[self.name], 
            is_streamed, 
            is_last
        )

        self.post(
            resp, 
            res, 
            resp.delta[self.name], 
            is_streamed, 
            is_last
        )
        return resp

    def prep(self) -> typing.Dict:
        return {}


class FromResp(Process):

    keys: typing.List[str]
    to_tuple: bool = True

    def forward(
        self, 
        resp: Resp
    ) -> typing.List[typing.Any] | typing.Dict[str, typing.Any]:

        if self.to_tuple:
            return tuple(
                resp.out[key] for key in self.keys
            )
        return {
            key: resp.out[key]
            for key in self.keys
        }
