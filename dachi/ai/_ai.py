# 1st party
import typing
from abc import ABC
from abc import ABC, abstractmethod
import typing

from .._core._core import (
    Module
)
from .._core import (
    Msg, Dialog, ListDialog
)

# PROMPT = typing.Union[Msg, Dialog]
# MESSAGE = typing.Union[ChatMsg, typing.List[ChatMsg]]

# LLM_RESPONSE = typing.Union[ChatMsg, typing.Tuple[ChatMsg, Dialog]]
LLM_PROMPT = typing.Union[typing.List[typing.Dict], typing.Dict]


def exclude_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:

    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    include = set(role)
    return [message for message in messages
        if message.role in include]


class LLM(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    def create_dialog(self, prompt):
        if isinstance(prompt, Dialog):
            return ListDialog([prompt])
        return prompt
    
    @abstractmethod
    def to_prompt(self, dialog: Dialog) -> typing.List[typing.Dict]:
        pass

    @abstractmethod
    def forward(
        self, prompt, 
        **kwarg_override
    ) -> Msg:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    def stream(
        self, prompt, 
        **kwarg_override
    ) -> typing.Iterator[Msg]:
        """API that allows for streaming the response

        Args:
            prompt (AIPrompt): Data to pass to the API

        Returns:
            typing.Iterator: Data representing the streamed response
            Uses 'delta' for the difference. Since the default
            behavior doesn't truly stream. This must be overridden 

        Yields:
            typing.Dict: The data
        """
        yield self.forward(prompt, **kwarg_override)
    
    async def aforward(
        self, prompt, **kwarg_override
    ) -> Msg:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(prompt, **kwarg_override)

    async def astream(
        self, prompt, **kwarg_override
    ) -> typing.AsyncIterator[Msg]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            prompt (AIPrompt): The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.forward(prompt, **kwarg_override)
        yield result

    def __call__(self, prompt: LLM_PROMPT, **kwarg_override) -> Msg:
        """Execute the AIModel

        Args:
            prompt (AIPrompt): The prompt

        Returns:
            AIResponse: Get the response from the AI
        """
        return self.forward(prompt, **kwarg_override)

    # def create_dialog(self, prompt: LLM_PROMPT):
    #     if isinstance(prompt, Dialog):
    #         return ListDialog([prompt])
    #     return prompt


# def exclude_type(messages: typing.Iterable[ChatMsg], *type_: str) -> typing.List[ChatMsg]:

#     exclude = set(type_)
#     filtered = []
#     for message in messages:
#         exclude_cur = False
#         for i in exclude:
#             exclude_cur = exclude_cur or isinstance(
#                 message, i
#             )
#         if exclude_cur:
#             filtered.append(message)
#     return filtered


# def include_type(messages: typing.Iterable[Msg], *type_: str) -> typing.List[Msg]:

#     exclude = set(type_)
#     filtered = []
#     for message in messages:
#         exclude_cur = False
#         for i in exclude:
#             exclude_cur = exclude_cur or isinstance(
#                 message, i
#             )
#         if exclude_cur:
#             filtered.append(message)
#     return filtered



# class EmbeddingModel(Module, ABC):
#     """APIAdapter allows one to adapt various WebAPI or otehr
#     API for a consistent interface
#     """

#     @abstractmethod
#     def forward(self, message: MESSAGE, **kwarg_override) -> ChatMsg:
#         """Run a standard query to the API

#         Args:
#             data : Data to pass to the API

#         Returns:
#             typing.Dict: The result of the API call
#         """
#         pass

#     def stream(
#         self, message: MESSAGE, **kwarg_override
#     ) -> typing.Iterator[ChatMsg]:
#         """API that allows for streaming the response

#         Args:
#             prompt (AIPrompt): Data to pass to the API

#         Returns:
#             typing.Iterator: Data representing the streamed response
#             Uses 'delta' for the difference. Since the default
#             behavior doesn't truly stream. This must be overridden 

#         Yields:
#             typing.Dict: The data
#         """
#         yield self.forward(message, **kwarg_override)
    
#     async def aforward(
#         self, message: MESSAGE, **kwarg_override
#     ) -> ChatMsg:
#         """Run this query for asynchronous operations
#         The default behavior is simply to call the query

#         Args:
#             data: Data to pass to the API

#         Returns:
#             typing.Any: 
#         """
#         return self.forward(message, **kwarg_override)

#     async def astream(
#         self, message: MESSAGE, **kwarg_override
#     ) -> typing.AsyncIterator[ChatMsg]:
#         """Run this query for asynchronous streaming operations
#         The default behavior is simply to call the query

#         Args:
#             prompt (AIPrompt): The data to pass to the API

#         Yields:
#             typing.Dict: The data returned from the API
#         """
#         result = self.forward(message, **kwarg_override)
#         yield result
    
#     def __call__(self, message: MESSAGE, **kwarg_override) -> ChatMsg:
#         """Execute the AIModel

#         Args:
#             prompt (AIPrompt): The prompt

#         Returns:
#             AIResponse: Get the response from the AI
#         """
#         return self.forward(message, **kwarg_override)


