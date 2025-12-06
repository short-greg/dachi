
# 1st party 
from abc import ABC, abstractmethod
import typing as t

# 3rd party
import pydantic

# local
from dachi.core import Inp
from ._process import Process


RESPONSE = 'resp'
S = t.TypeVar('S', bound=pydantic.BaseModel)

class ToPrompt(Process, ABC):
    """Converts the input to a message

    Example:
        to_msg = ToText(role='user', field='text')
        msg = to_msg("Hello, world!")
        print(msg)
        # Msg(role='user', text='Hello, world!')
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Inp:
        """Convert the args and kwargs to a prompt

        Returns:
            Msg: A prompt message
        """
        pass


class NullToPrompt(ToPrompt):
    """Converts a message to a prompt (wraps in Prompt if not already)

    Example:
        to_prompt = NullToPrompt()
        prompt = to_prompt(Msg(role='user', text='Hello, world!'))
        print(prompt)
    """

    def forward(self, msg: Inp) -> Inp:
        """Convert a message to a prompt

        Returns:
            Msg: A prompt message
        """
        return msg

# TO_PROMPT = t.TypeVar("TO_PROMPT", bound=ToPrompt)

# class NullToPrompt(ToPrompt):
#     """Converts a message to a prompt (wraps in Prompt if not already)

#     Example:
#         to_prompt = NullToPrompt()
#         prompt = to_prompt(Msg(role='user', text='Hello, world!'))
#         print(prompt)
#         # Prompt(role='user', text='Hello, world!')
#     """

#     def forward(self, msg: Msg) -> Prompt:
#         """Convert a message to a prompt

#         Returns:
#             Prompt: A prompt message
#         """
#         if isinstance(msg, Prompt):
#             return msg
#         # Convert Msg to Prompt
#         return Prompt(**msg.model_dump())


# class ToText(ToPrompt):
#     """Converts the input to a text message

#     Example:
#         to_msg = ToText(role='user', field='text')
#         msg = to_msg("Hello, world!")
#         print(msg)
#         # Msg(role='user', text='Hello, world!')
#     Args:
#         role (str, optional): The role of the message. Defaults to 'system'.
#         field (str, optional): The field to use for the text. Defaults to 'content
#     """
#     role: str = 'system'
#     field: str = 'content'

#     def forward(self, text: str) -> Prompt:
#         """Create a text prompt

#         Args:
#             text (str): The text for the message

#         Returns:
#             Prompt: Converts to a text prompt
#         """
#         return Prompt(
#             role=self.role,
#             **{self.field: text}
#         )
