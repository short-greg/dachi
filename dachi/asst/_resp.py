# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..msg._messages import Msg, END_TOK, StreamMsg
from ._msg import MsgConv
from .. import utils

RESPONSE = 'resp'


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class RespConv(MsgConv, ABC):
    """Use to process the resoponse from an LLM
    """

    conv_env_tok = ''

    def __init__(self, name):
        super().__init__(name, 'response')

    def handle_end_tok(self, resp):

        if resp is END_TOK:
            return self.conv_env_tok
        return resp
    
    @abstractmethod
    def delta(
        self, response, msg: Msg, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False
    ) -> Msg: 
        pass

    def forward(self, msg: Msg):
        resp = msg['meta']['response']
        if isinstance(msg, StreamMsg):
            streamed = True
            is_last = msg.is_last
        else:
            streamed = False
            is_last = True
        if resp is END_TOK:
            resp = self.handle_end_tok(resp)
        
        if resp is utils.UNDEFINED:
            return utils.UNDEFINED
        msg['meta'][self.name] = self.delta(
            resp, {}, streamed, is_last
        )
        return msg

    def prep(self) -> typing.Dict:
        return {}
