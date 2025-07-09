
# 1st party 
from abc import ABC, abstractmethod
import typing

# local
from ..core import Msg, Resp
from ._process import Process
from .. import utils


RESPONSE_FIELD = 'response'


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
        if isinstance(self.from_, str):
            from_ = [self.from_]
        else:
            from_ = self.from_
        r = [resp.data[r] for r in from_]
        is_undefined = all(r is utils.UNDEFINED for r in r)
        
        if self._single:
            r = r[0]
        
        if is_undefined:
            resp.data[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        
        # delta_store = utils.get_or_set(
        #     r.delta, self.name, {}
        # )
        resp.data[self.name] = res = self.delta(
            r, delta_store, is_streamed, is_last
        )
        self.post(resp, res, delta_store, is_streamed, is_last)
        return resp

    @classmethod
    def run(
        cls, 
        resp: Resp, 
        proc: typing.Union['RespProc', None, typing.List['RespProc']],
        is_streamed: bool=False, is_last: bool=True
    ) -> 'Resp':
        if proc is None:
            return resp
        if isinstance(proc, RespProc):
            return proc(resp, is_streamed, is_last)
        
        for p in proc:
            resp = p(resp, is_streamed, is_last)
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


# class RespConv(Process, ABC):
#     """Use to process the resoponse from an LLM
#     """

#     name: str

#     @abstractmethod
#     def delta(
#         self, 
#         resp, 
#         delta_store: typing.Dict, 
#         streamed: bool=False, 
#         is_last: bool=False
#     ) -> Msg: 
#         pass

#     def forward(
#         self, 
#         resp: Resp, 
#         is_streamed: bool=False, 
#         is_last: bool=True
#     ):

#         r = resp.val

#         delta_store = utils.get_or_set(
#             resp.delta, self.name, {}
#         )
        
#         resp.data[self.name] = res = self.delta(
#             r, 
#             delta_store, 
#             is_streamed, 
#             is_last
#         )

#         self.post(
#             resp, 
#             res, 
#             resp.delta[self.name], 
#             is_streamed, 
#             is_last
#         )
#         return resp

#     def prep(self) -> typing.Dict:
#         return {}


class FromResp(Process):

    keys: typing.List[str] | str
    as_dict: bool = False

    def forward(
        self, 
        resp: Resp
    ) -> typing.List[typing.Any] | typing.Dict[str, typing.Any]:
        if not self.as_dict:
            if isinstance(self.keys, str):
                return resp.out[self.keys]
            return tuple(
                resp.out[key] 
                for key in self.keys
            )
    
        if isinstance(self.keys, str):
            keys = [self.keys]
        else:
            keys = self.keys
        return {
            key: resp.out[key]
            for key in keys
        }
