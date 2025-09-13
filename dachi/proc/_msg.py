
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

    Example:
        to_msg = ToText(role='user', field='text')
        msg = to_msg("Hello, world!")
        print(msg)
        # Msg(role='user', text='Hello, world!')
    """

    @abstractmethod
    def delta(self, *args, **kwargs) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        pass


class NullToMsg(ToMsg):
    """Converts a message to a message (so actually does nothing)

    Example:
        to_msg = NullToMsg()
        msg = to_msg(Msg(role='user', text='Hello, world!'))
        print(msg)
        # Msg(role='user', text='Hello, world!')
    """

    def delta(self, msg: Msg) -> Msg:
        """Convert the args and kwargs to a message

        Returns:
            Msg: A message
        """
        return msg


class ToText(ToMsg):
    """Converts the input to a text message

    Example:
        to_msg = ToText(role='user', field='text')
        msg = to_msg("Hello, world!")
        print(msg)
        # Msg(role='user', text='Hello, world!')
    Args:
        role (str, optional): The role of the message. Defaults to 'system'.
        field (str, optional): The field to use for the text. Defaults to 'content
    """
    role: str = 'system'
    field: str = 'content'

    def delta(self, text: str) -> Msg:
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
