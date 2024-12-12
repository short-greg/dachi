# 1st party
import typing
from abc import ABC
from abc import ABC, abstractmethod
import typing

from .._core._core import (
    Module
)
from .._core import Message, Dialog



PROMPT = typing.Union[Message, Dialog]
MESSAGE = typing.Union[Message, typing.List[Message]]

class EmbeddingModel(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    @abstractmethod
    def forward(self, message: MESSAGE, **kwarg_override) -> Message:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    def stream(self, message: MESSAGE, **kwarg_override) -> typing.Iterator[Message]:
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
        yield self.forward(message, **kwarg_override)
    
    async def aforward(
        self, message: MESSAGE, **kwarg_override
    ) -> Message:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(message, **kwarg_override)

    async def astream(
        self, message: MESSAGE, **kwarg_override
    ) -> typing.AsyncIterator[Message]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            prompt (AIPrompt): The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.forward(message, **kwarg_override)
        yield result

    # async def _collect_results(generator, index, results, queue):
        
    #     async for item in generator:
    #         results[index] = item
    #         await queue.put(results[:])  # Put a copy of the current results
    #     results[index] = None  # Mark this generator as completed

    def __call__(self, message: MESSAGE, **kwarg_override) -> Message:
        """Execute the AIModel

        Args:
            prompt (AIPrompt): The prompt

        Returns:
            AIResponse: Get the response from the AI
        """
        return self.forward(message, **kwarg_override)


class LLModel(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    @abstractmethod
    def forward(self, prompt: typing.Union[Message, Dialog], **kwarg_override) -> Message:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    def stream(self, prompt: typing.Union[Message, Dialog], **kwarg_override) -> typing.Iterator[Message]:
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
        self, prompt: typing.Union[Message, Dialog], **kwarg_override
    ) -> Message:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(prompt, **kwarg_override)

    async def astream(
        self, prompt: typing.Union[Message, Dialog], **kwarg_override
    ) -> typing.AsyncIterator[Message]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            prompt (AIPrompt): The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.forward(prompt, **kwarg_override)
        yield result

    async def _collect_results(generator, index, results, queue):
        
        async for item in generator:
            results[index] = item
            await queue.put(results[:])  # Put a copy of the current results
        results[index] = None  # Mark this generator as completed

    def __call__(self, prompt: typing.Union[Message, Dialog], **kwarg_override) -> Message:
        """Execute the AIModel

        Args:
            prompt (AIPrompt): The prompt

        Returns:
            AIResponse: Get the response from the AI
        """
        return self.forward(prompt, **kwarg_override)

    # async def bulk_async_forward(
    #     self, prompt: typing.List[AIPrompt], **kwarg_override
    # ) -> typing.List[AIResponse]:
    #     """

    #     Args:
    #         messages (typing.List[typing.List[Message]]): 

    #     Returns:
    #         typing.List[typing.Dict]: 
    #     """
    #     tasks = []
    #     async with asyncio.TaskGroup() as tg:

    #         for prompt_i in prompt:
    #             tasks.append(
    #                 tg.create_task(self.aforward(prompt_i, **kwarg_override))
    #             )
    #     return list(
    #         task.result() for task in tasks
    #     )

    # async def bulk_async_stream_forward(
    #     self, prompts: typing.List[AIPrompt], **kwarg_override
    # ) -> typing.AsyncIterator[typing.List[AIResponse]]:
    #     """Process multiple 

    #     Args:
    #         prompts (AIPrompt): The prompts to process

    #     Returns:
    #         typing.List[typing.Dict]: 
    #     """
    #     results = [None] * len(prompts)
    #     queue = asyncio.Queue()

    #     async with asyncio.TaskGroup() as tg:
    #         for index, prompt_i in enumerate(prompts):
    #             tg.create_task(self._collect_results(
    #                 self.astream(prompt_i, **kwarg_override), index, results, queue)
    #             )

    #     active_generators = len(prompts)
    #     while active_generators > 0:
    #         current_results = await queue.get()
    #         yield current_results
    #         active_generators = sum(result is not None for result in current_results)

    # @abstractmethod
    # def convert(self, message: Message) -> typing.Dict:
    #     """Convert a message to the format needed for the model

    #     Args:
    #         messages (Message): The messages to convert

    #     Returns:
    #         typing.List[typing.Dict]: The format to pass to the "model"
    #     """
    #     pass

    # def convert_messages(self, messages: typing.List[Message]) -> typing.List[typing.Dict]:
    #     """Convienence method to convert a list of messages to the format needed for the model

    #     Args:
    #         messages (typing.List[Message]): The messages to convert

    #     Returns:
    #         typing.List[typing.Dict]: The format to pass to the "model"
    #     """
    #     return [self.convert(message) for message in messages]


# @dataclass
# class AIResponse(object):
#     """The AI response stores the response from the API
#     And any processing done on it.
#     """

#     message: 'Message' # The message from the AI
#     source: typing.Dict # The response from the API
#     val: typing.Any = None # the result of processing the message

#     def clone(self) -> Self:

#         return AIResponse(
#             message=self.message,
#             source=self.source,
#             val=self.val
#         )
    
#     def __iter__(self) -> typing.Iterator:
#         """Unpack the values in the AIResponse

#         Yields:
#             Iterator[typing.Iterator]: Each value in the response
#         """
#         yield self.source
#         yield self.message
#         yield self.val


# class AIPrompt(pydantic.BaseModel, ABC):
#     """Base class for prompts to send to the AI
#     """

#     @abstractmethod
#     def prompt(self, model: 'AIModel'):
#         """Send the prompt to the AI Model

#         Args:
#             model (AIModel): The model to query
#         """
#         pass

#     @abstractmethod
#     def instruct(self, cue: 'Cue', model: 'AIModel'):
#         """Send an instruction to the model

#         Args:
#             cue (Cue): The cue to use
#             model (AIModel): The model to use
#         """
#         pass

#     @abstractmethod
#     def reader(self) -> 'Reader':
#         """Get the reader from the AIPrompt to process the response

#         Returns:
#             Reader: The reader used to compute val in the response
#         """
#         pass

#     @abstractmethod
#     def process_response(self, response: 'AIResponse') -> 'AIResponse':
#         """Process the response of the Model

#         Args:
#             response (AIResponse): The response to process

#         Returns:
#             AIResponse: The updated response
#         """
#         pass

#     @abstractmethod
#     def clone(self) -> typing.Self:
#         """Do a shallow clone of the message object

#         Returns:
#             typing.Self: The cloned object
#         """
#         pass

#     @abstractmethod
#     def aslist(self) -> typing.List['Message']:
#         """Return the AIPrompt as a list of messages
#         """
#         pass

#     def __iter__(self) -> typing.Iterator['Message']:
#         """Iterate over each message in the prompt

#         Yields:
#             Message: Each message
#         """
#         for message in self.aslist():
#             yield message

# def stream_text(stream: typing.Iterator[typing.Tuple[AIResponse, AIResponse]]) -> typing.Iterator[TextMessage]:
#     """Stream the text response from an AI Engine

#     Args:
#         stream (typing.Iterator[typing.Tuple[AIResponse, AIResponse]]): The stream to execute

#     Returns:
#         typing.Iterator[TextMessage]: 

#     Yields:
#         Iterator[typing.Iterator[TextMessage]]: 
#     """
    
#     for _, a2 in stream:
#         yield a2.message



# Data = typing.Union[pydantic.BaseModel, typing.List[pydantic.BaseModel]]


# class AIModel(Module, ABC):
#     """APIAdapter allows one to adapt various WebAPI or otehr
#     API for a consistent interface
#     """

#     @abstractmethod
#     def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
#         """Run a standard query to the API

#         Args:
#             data : Data to pass to the API

#         Returns:
#             typing.Dict: The result of the API call
#         """
#         pass

#     @abstractmethod
#     def convert(self, message: Message) -> typing.Dict:
#         """Convert a message to the format needed for the model

#         Args:
#             messages (Message): The messages to convert

#         Returns:
#             typing.List[typing.Dict]: The format to pass to the "model"
#         """
#         pass

#     def convert_messages(self, messages: typing.List[Message]) -> typing.List[typing.Dict]:
#         """Convienence method to convert a list of messages to the format needed for the model

#         Args:
#             messages (typing.List[Message]): The messages to convert

#         Returns:
#             typing.List[typing.Dict]: The format to pass to the "model"
#         """
#         return [self.convert(message) for message in messages]

#     def stream(self, prompt: AIPrompt, **kwarg_override) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:
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
#         result = self.forward(prompt, **kwarg_override)
#         yield result, result
    
#     async def aforward(
#         self, prompt: AIPrompt, **kwarg_override
#     ) -> AIResponse:
#         """Run this query for asynchronous operations
#         The default behavior is simply to call the query

#         Args:
#             data: Data to pass to the API

#         Returns:
#             typing.Any: 
#         """
#         return self.forward(prompt, **kwarg_override)
    
#     async def bulk_async_forward(
#         self, prompt: typing.List[AIPrompt], **kwarg_override
#     ) -> typing.List[AIResponse]:
#         """

#         Args:
#             messages (typing.List[typing.List[Message]]): 

#         Returns:
#             typing.List[typing.Dict]: 
#         """
#         tasks = []
#         async with asyncio.TaskGroup() as tg:

#             for prompt_i in prompt:
#                 tasks.append(
#                     tg.create_task(self.aforward(prompt_i, **kwarg_override))
#                 )
#         return list(
#             task.result() for task in tasks
#         )
    
#     async def astream(
#         self, prompt: AIPrompt, **kwarg_override
#     ) -> typing.AsyncIterator[AIResponse]:
#         """Run this query for asynchronous streaming operations
#         The default behavior is simply to call the query

#         Args:
#             prompt (AIPrompt): The data to pass to the API

#         Yields:
#             typing.Dict: The data returned from the API
#         """
#         result = self.forward(prompt, **kwarg_override)
#         yield result

#     async def _collect_results(generator, index, results, queue):
        
#         async for item in generator:
#             results[index] = item
#             await queue.put(results[:])  # Put a copy of the current results
#         results[index] = None  # Mark this generator as completed

#     async def bulk_async_stream_forward(
#         self, prompts: typing.List[AIPrompt], **kwarg_override
#     ) -> typing.AsyncIterator[typing.List[AIResponse]]:
#         """Process multiple 

#         Args:
#             prompts (AIPrompt): The prompts to process

#         Returns:
#             typing.List[typing.Dict]: 
#         """
#         results = [None] * len(prompts)
#         queue = asyncio.Queue()

#         async with asyncio.TaskGroup() as tg:
#             for index, prompt_i in enumerate(prompts):
#                 tg.create_task(self._collect_results(
#                     self.astream(prompt_i, **kwarg_override), index, results, queue)
#                 )

#         active_generators = len(prompts)
#         while active_generators > 0:
#             current_results = await queue.get()
#             yield current_results
#             active_generators = sum(result is not None for result in current_results)

#     def __call__(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
#         """Execute the AIModel

#         Args:
#             prompt (AIPrompt): The prompt

#         Returns:
#             AIResponse: Get the response from the AI
#         """
#         return self.forward(prompt, **kwarg_override)
