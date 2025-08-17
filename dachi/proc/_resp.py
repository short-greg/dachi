
# 1st party 
from abc import ABC, abstractmethod
import typing
import json
import inspect

# 3rd party
import pydantic

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
            resp.delta.proc_store, self.name, {}
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


class TextConv(RespProc):
    """
    Unified text processor that extracts text content from responses.
    Works with the unified Resp structure instead of OpenAI-specific format.
    """
    name: str = 'content'
    from_: str = 'text'

    def post(
        self, 
        resp: Resp, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = False
    ):
        """Update the message content with accumulated text."""
        content = delta_store.get('content', '')
        if resp.msg:
            resp.msg.text = content
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
        """Process text content from unified response structure."""
        if streamed and is_last:
            return ''
        
        if resp is None or resp == '':
            content = ''
        else:
            content = str(resp)

        delta_store['cur_content'] = content
        delta_store['content'] = utils.acc(
            delta_store, 'all_content', content
        )

        return content


class StructConv(RespProc):
    """
    Unified structured data converter for JSON/structured outputs.
    Works with the unified Resp structure.
    """
    struct: typing.Union[pydantic.BaseModel, typing.Dict, None] = None
    name: str = 'content'
    from_: str = 'text'

    def post(
        self, 
        resp: Resp, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = False
    ):
        """Update message content with structured data."""
        if is_last and resp.msg:
            resp.msg.text = delta_store.get('content', '')
        elif resp.msg:
            resp.msg.text = ''

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
        """Process structured data from unified response."""
        if streamed and resp is not None:
            delta_content = str(resp) if resp else ''
            utils.acc(delta_store, 'content', delta_content)
        elif not streamed and resp is not None:
            content = str(resp) if resp else ''
            utils.acc(delta_store, 'content', content)

        if is_last:
            try:
                struct = json.loads(delta_store.get('content', '{}'))
                return struct
            except json.JSONDecodeError:
                return {}

        return ''

    def prep(self) -> typing.Dict:
        """Prepare request parameters for structured output."""
        if isinstance(self.struct, typing.Dict):
            return {'response_format': {
                "type": "json_schema",
                "json_schema": self.struct
            }}
        elif isinstance(self.struct, pydantic.BaseModel) or (
            inspect.isclass(self.struct) and 
            issubclass(self.struct, pydantic.BaseModel)
        ):
            return {'response_format': self.struct}
        return {'response_format': "json_object"}


class ParsedConv(RespProc):
    """
    Unified parsed data converter that validates JSON against Pydantic models.
    Works with the unified Resp structure.
    """
    struct: pydantic.BaseModel = None
    name: str = 'content'
    from_: str = 'text'
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
        """Parse and validate structured data against Pydantic model."""
        if not is_last:
            return utils.UNDEFINED
            
        content = str(resp) if resp else '{}'
        try:
            delta_store["content"] = self.struct.model_validate_json(content)
            return delta_store["content"]
        except Exception:
            return utils.UNDEFINED
    
    def prep(self):
        """Prepare request parameters for structured output with schema."""
        return {
            "response_format": {
                "type": "json_schema", 
                "json_schema": {
                    "name": self.struct.__name__,
                    "schema": self.struct.model_json_schema()
                }
            }
        }
