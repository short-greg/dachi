# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ._messages import Msg, StreamMsg
from ._msg import MsgProc

RESPONSE = 'resp'


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class RespConv(MsgProc, ABC):
    """Use to process the resoponse from an LLM
    """

    conv_env_tok = lambda : {'content': ''}

    def __init__(self, name, from_: str='response'):
        super().__init__(name, from_)

    @abstractmethod
    def delta(
        self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False
    ) -> Msg: 
        pass

    def forward(self, msg: Msg, delta_store: typing.Dict=None):

        if delta_store is None:
            delta_store = {}
        resp = msg['meta'][self._from[0]]

        if isinstance(msg, StreamMsg):
            streamed = True
            is_last = msg.is_last
        else:
            streamed = False
            is_last = True
        
        msg['meta'][self.name] = res = self.delta(
            resp, delta_store, streamed, is_last
        )

        self.post(msg, res, delta_store, streamed, is_last)
        return msg

    def prep(self) -> typing.Dict:
        return {}
