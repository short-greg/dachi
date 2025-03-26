# 1st party
import typing
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from ..msg._messages import Msg


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class RespConv(ABC):
    """Use to process the resoponse from an LLM
    """
    def __init__(self, resp: bool):
        """
        Initialize the instance.
        Args:
            resp (bool): Indicates if the response processor responds with data.
        """
        super().__init__()
        self._resp = resp

    @property
    def resp(self) -> bool:
        """Choose whether to include a response

        Returns:
            bool: Whether to respond with a value
        """
        return self._resp

    @abstractmethod
    def __call__(self, response, msg: Msg) -> typing.Any:
        pass

    @abstractmethod
    def delta(
        self, response, msg: Msg, delta_store: typing.Dict
    ) -> typing.Any: 
        pass

    def prep(self) -> typing.Dict:
        return {}
