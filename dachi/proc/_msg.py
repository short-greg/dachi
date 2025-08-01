
# 1st party 
from abc import ABC, abstractmethod
import typing

# 3rd party
import pydantic

# local
from dachi.core import Msg
from ._process import Process

RESPONSE = 'resp'
S = typing.TypeVar('S', bound=pydantic.BaseModel)


class ToMsg(Process, ABC):
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
