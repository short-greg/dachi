# 1st party
import typing
from abc import ABC, abstractmethod
from typing import Self
import typing

import asyncio
from ._core import (
    Instruct, Instruction, Reader, 
    render, NullRead, Module
)
from dataclasses import dataclass
from ._core import Struct

# 3rd party
import pydantic


@dataclass
class AIResponse(object):

    message: 'Message'
    source: typing.Dict
    val: typing.Any = None

    def clone(self) -> Self:

        return AIResponse(
            message=self.message,
            source=self.source,
            val=self.val
        )
    
    def __iter__(self) -> typing.Iterator:
        yield self.source
        yield self.message
        yield self.val


class AIPrompt(Struct, ABC):

    @abstractmethod
    def prompt(self, model: 'AIModel'):
        pass

    @abstractmethod
    def instruct(self, instruction: 'Instruction', model: 'AIModel'):
        pass

    @abstractmethod
    def reader(self) -> 'Reader':
        pass

    @abstractmethod
    def process_response(self, response: 'AIResponse') -> 'AIResponse':
        """

        Args:
            response (AIResponse): The response to process

        Returns:
            AIResponse: The updated response
        """
        pass

    @abstractmethod
    def clone(self) -> typing.Self:
        """Do a shallow clone of the message object

        Returns:
            typing.Self: The cloned object
        """
        pass

    @abstractmethod
    def aslist(self) -> typing.List['Message']:
        """
        """
        pass

    def __iter__(self) -> typing.Iterator['Message']:
        """

        Yields:
            Message: Each message
        """
        for message in self.aslist():
            yield message

    def __call__(self, prompt: 'AIPrompt') -> 'AIResponse':

        return self.forward(prompt)


class Message(Struct):

    source: str
    data: typing.Dict[str, typing.Any]

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f'{key}')

    def __setitem__(self, key: str, value: typing.Any):
        if hasattr(self, key):
            setattr(self, key, value)
        if key in self.data:
            self.data[key] = value
        raise KeyError(f'{key}')

    def prompt(self, model: 'AIModel', **kwarg_overrides) -> 'AIResponse':
        return model(self, **kwarg_overrides)

    def reader(self) -> 'Reader':
        return NullRead()

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            data=self.data
        )

    def aslist(self) -> typing.List['Message']:
        return [self]
    
    def render(self) -> str:
        """Render the message

        Returns:
            str: Return the message and the source
        """
        return f'{self.source}: {render(self.data)}'


class TextMessage(Message):

    def __init__(self, source: str, text: typing.Union[str, 'Instruction']) -> 'Message':

        super().__init__(
            source=source,
            data={
                'text': text
            }
        )

    def reader(self) -> 'Reader':
        text = self['text']
        if isinstance(text, Instruction) and text.out is not None:
            return text.out
        return NullRead(name='')
    
    def render(self) -> str:
        """Render the text message

        Returns:
            str: Return the message and the text for the message
        """
        text = self.data['text']
        return f'{self.source}: {
            text.render() if isinstance(text, Instruction) else text
        }'

    def clone(self) -> typing.Self:
        """Do a shallow copy of the message

        Returns:
            Message: The cloned message
        """
        return self.__class__(
            source=self.source,
            text=self.data['text']
        )

    @property
    def text(self) -> str:
        return self.data['text']


def stream_text(stream: typing.Iterator[typing.Tuple[AIResponse, AIResponse]]) -> typing.Iterator[TextMessage]:
    
    for _, a2 in stream:
        yield a2.message


class Dialog(Struct):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

    messages: typing.List[Message] = pydantic.Field(default_factory=list)

    def __iter__(self) -> typing.Iterator[Message]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Message]]: Each message in the dialog
        """

        for message in self.messages:
            yield message

    def __add__(self, other: 'Dialog') -> 'Dialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        return Dialog(
            self.messages + other.messages
        )

    def __getitem__(self, idx) -> Message:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Message: The message in the dialog
        """
        return self.messages[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        self.messages[idx] = message
        return self

    def insert(self, index: int, message: Message):
        """Insert a value into the dialog

        Args:
            index (int): The index to insert at
            message (Message): The message to insert
        """
        self.messages.insert(index, message)

    def pop(self, index: int):
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        self.messages.pop(index)

    def remove(self, message: Message):
        """Remove a message from the dialog

        Args:
            message (Message): The message to remove
        """
        self.messages.remove(message)

    def append(self, message: Message):
        """Append a message to the end of the dialog

        Args:
            message (Message): The message to add
        """
        self.messages.append(message)

    def add(self, message: Message, ind: typing.Optional[int]=None, replace: bool=False):
        """Add a message to the dialog

        Args:
            message (Message): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if ind is not None and ind < 0:
            ind = max(len(self.messages) + ind, 0)

        if ind is None or ind == len(self.messages):
            if not replace or ind == len(self.messages):
                self.messages.append(message)
            else:
                self.messages[-1] = message
        elif ind > len(self.messages):
            raise ValueError(
                f'The index {ind} is out of bounds '
                f'for size {len(self.messages)}')
        elif replace:
            self.messages[ind] = message
        else:
            self.messages.insert(ind, message)
        
    def clone(self) -> 'Dialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        return Dialog(
            messages=[message.clone() for message in self.messages]
        )

    def extend(self, dialog: typing.Union['Dialog', typing.List[Message]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Message]]): _description_
        """
        if isinstance(dialog, Dialog):
            dialog = dialog.messages
        
        self.messages.extend(dialog)

    def message(self, source: str, text: typing.Optional[str]=None, _ind: typing.Optional[int]=None, _replace: bool=False, **kwargs):
        """Add a message to the 

        Args:
            source (str): the source of the message
            text (typing.Optional[str], optional): The text message. Defaults to None.
            _ind (typing.Optional[int], optional): The index to set at. Defaults to None.
            _replace (bool, optional): Whether to replace the the text at the index. Defaults to False.

        Raises:
            ValueError: If no message was passed in
        """
        if len(kwargs) == 0 and text is not None:
            message = TextMessage(source, text)
        elif text is not None:
            message = Message(text=text, **kwargs)
        elif text is None:
            message = Message(**kwargs)
        else:
            raise ValueError('No message has been passed. The text and kwargs are empty')

        self.add(message, _ind, _replace)

    def user(self, text: str=None, _ind: int=None, _replace: bool=False, **kwargs):
        """Add a user message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('user', text, _ind, _replace, **kwargs)

    def assistant(self, text: str=None, _ind=None, _replace: bool=False, **kwargs):
        """Add an assistant message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('assistant', text, _ind, _replace, **kwargs)

    def system(self, text: str=None, _ind=None, _replace: bool=False, **kwargs):
        """Add a system message

        Args:
            text (str, optional): The text for the message. Defaults to None.
            _ind (int, optional): The index to add to. Defaults to None.
            _replace (bool, optional): Whether to replace at the index. Defaults to False.
        """
        self.message('system', text, _ind, _replace, **kwargs)

    def reader(self) -> 'Reader':
        """Get the "Reader" for the dialog. By default will use the last one
        that is available.

        Returns:
            Reader: The reader to retrieve
        """
        for r in reversed(self.messages):
            if isinstance(r, Instruction):
                if r.reader is not None:
                    return r.reader
        return NullRead(name='')
    
    def exclude(self, *source: str) -> 'Dialog':

        exclude = set(source)
        return Dialog(
            messages=[message for message in self.messages
            if message.source not in exclude]
        )
    
    def include(self, *source: str) -> 'Dialog':
        include = set(source)
        return Dialog(
            messages=[message for message in self.messages
            if message.source in include]
        )

    def render(self) -> str:
        return '\n'.join(
            message.render() for message in self.messages
        )

    def instruct(
        self, instruct: 'Instruct', ai_model: 'AIModel', 
        ind: int=0, replace: bool=True
    ) -> AIResponse:
        """Instruct the AI

        Args:
            instruct (Instruct): The instruction to use
            ai_model (AIModel): The AIModel to use
            ind (int, optional): The index to set to. Defaults to 0.
            replace (bool, optional): Whether to replace at the index if already set. Defaults to True.

        Returns:
            AIResponse: The output from the AI
        """
        instruction = instruct.i()
        
        self.system(instruction, ind, replace)
        response = ai_model.forward(self.messages)
        response = self.process_response(response)
        self.assistant(response.content)
        return response

    def prompt(self, model: 'AIModel', append: bool=True) -> 'AIResponse':
        """Prompt the AI

        Args:
            model (AIModel): The model to usee
            append (bool, optional): Whether to append the output. Defaults to True.

        Returns:
            AIResponse: The response from the AI
        """
        response = model(self)

        if append:
            self.append(response.message)
        return response

    def stream_prompt(self, model: 'AIModel', append: bool=True, **kwarg_override) -> typing.Iterator[typing.Tuple['AIResponse', 'AIResponse']]:
        """Prompt the AI

        Args:
            model (AIModel): The model to usee
            append (bool, optional): Whether to append the output. Defaults to True.

        Returns:
            AIResponse: The response from the AI
        """
        for d, dx in model.stream_forward(self, **kwarg_override):
            yield d, dx

        if append:
            self.append(d.message)
        return d

    def aslist(self) -> typing.List['Message']:
        """Retrieve the message list

        Returns:
            typing.List[Message]: the messages in the dialog
        """
        return self.messages
    
    def __len__(self) -> int:
        return len(self.messages)


Data = typing.Union[Struct, typing.List[Struct]]


class AIModel(Module, ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    @abstractmethod
    def forward(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    @abstractmethod
    def convert(self, message: Message) -> typing.Dict:
        """Convert a message to the format needed for the model

        Args:
            messages (Message): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        pass

    def convert_messages(self, messages: typing.List[Message]) -> typing.List[typing.Dict]:
        """Convienence method to convert a list of messages to the format needed for the model

        Args:
            messages (typing.List[Message]): The messages to convert

        Returns:
            typing.List[typing.Dict]: The format to pass to the "model"
        """
        return [self.convert(message) for message in messages]

    def stream_forward(self, prompt: AIPrompt, **kwarg_override) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:
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
        result = self.forward(prompt, **kwarg_override)
        yield result, result
    
    async def async_forward(
        self, prompt: AIPrompt, **kwarg_override
    ) -> AIResponse:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.forward(prompt, **kwarg_override)
    
    async def bulk_async_forward(
        self, prompt: typing.List[AIPrompt], **kwarg_override
    ) -> typing.List[AIResponse]:
        """

        Args:
            messages (typing.List[typing.List[Message]]): 

        Returns:
            typing.List[typing.Dict]: 
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:

            for prompt_i in prompt:
                tasks.append(
                    tg.create_task(self.async_forward(prompt_i, **kwarg_override))
                )
        return list(
            task.result() for task in tasks
        )
    
    async def async_stream_forward(self, prompt: AIPrompt, **kwarg_override) -> typing.AsyncIterator[AIResponse]:
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

    async def bulk_async_stream_forward(
        self, prompts: typing.List[AIPrompt], **kwarg_override
    ) -> typing.AsyncIterator[typing.List[AIResponse]]:
        """Process multiple 

        Args:
            prompts (AIPrompt): The prompts to process

        Returns:
            typing.List[typing.Dict]: 
        """
        results = [None] * len(prompts)
        queue = asyncio.Queue()

        async with asyncio.TaskGroup() as tg:
            for index, prompt_i in enumerate(prompts):
                tg.create_task(self._collect_results(
                    self.async_stream_forward(prompt_i, **kwarg_override), index, results, queue)
                )

        active_generators = len(prompts)
        while active_generators > 0:
            current_results = await queue.get()
            yield current_results
            active_generators = sum(result is not None for result in current_results)

    def __call__(self, prompt: AIPrompt, **kwarg_override) -> AIResponse:
        return self.forward(prompt, **kwarg_override)


    # def process_response(self, response: AIResponse) -> AIResponse:

    #     response = response.clone()

    #     out = None
    #     for r in reversed(self.messages):
    #         if  isinstance(r, Instruction):
    #             out = r.out
    #             break

    #     if out is None:
    #         response.val = response.content
    #     else:
    #         # Not sure about this
    #         response.val = out.read(response.content['text'])
    #     return response

