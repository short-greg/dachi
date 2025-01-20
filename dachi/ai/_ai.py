# 1st party
import typing
import json
from abc import ABC, abstractmethod
import typing
from .._core._core import (
    Module
)
import pydantic
from .._core import Msg, ListDialog, Dialog
from ..utils._f_utils import (
    is_async_function, is_async_function, 
    is_generator_function
)

LLM_PROMPT = typing.Union[typing.Iterable[Msg], Msg]
LLM_RESPONSE = typing.Tuple[Msg, typing.Any]


class ToolOption(pydantic.BaseModel):
    """Create an option for a tool to pass to the model
    """
    name: str
    f: typing.Callable[[typing.Any], typing.Any]
    # required: typing.List[str] = pydantic.Field(default_factory=list)
    # params: typing.List[typing.Dict]
    kwargs: typing.Dict

    def to_input(self) -> typing.Dict:

        return {
            # 'name': self.name,
            # 'required': self.required,
            **self.kwargs
        }


class ToolSet(object):
    """A set of tools that the LLM can use
    """
    def __init__(self, tools: typing.List[ToolOption], **kwargs):
        """The set of tools

        Args:
            tools (typing.List[ToolOption]): The list of tools
        """
        self.tools = {
            tool.name: tool
            for tool in tools
        }
        self.kwargs = kwargs

    def add(self, option: ToolOption):
        """Add a tool to the set

        Args:
            option (ToolOption): The option to add
        """
        self.tools[option.name] = option

    def remove(self, option: ToolOption):
        """Remove a tool from the tool set

        Args:
            option (ToolOption): The option to add
        """
        del self.tools[option.name]

    def to_input(self):
        return list(
            tool.to_input() for _, tool in self.tools.items()
        )
    
    def __iter__(self) -> typing.Iterator:

        for _, tool in self.tools.items():
            yield tool

    def __getitem__(self, name):
        return self.tools[name]


class ToolCall(pydantic.BaseModel):
    """A response from the LLM that a tool was called
    """
    option: ToolOption = pydantic.Field(
        description="The tool that was chosen."
    )
    args: typing.Dict[str, typing.Any] = pydantic.Field(
        description="The arguments to the tool."
    )

    def forward(self) -> typing.Any:
        """Call the tool

        Raises:
            NotImplementedError: If the function is async
            NotImplementedError: If the function is a generator function

        Returns:
            typing.Any: The result of the call
        """
        # Check if valid to use with forward
        if is_async_function(self.option.f):
            raise NotImplementedError
        if is_generator_function(self.option.f):
            raise NotImplementedError
        return self.option.f(**self.args)

    async def aforward(self) -> typing.Any:
        """Call the tool 

        Raises:
            NotImplementedError: If the function is a generator

        Returns:
            typing.Any: The result of the call
        """
        if is_async_function(self.option.f):
            return await self.option.f(**self.args)
        if is_generator_function(self.option.f):
            raise NotImplementedError
        return self.option.f(**self.args)

    def stream(self) -> typing.Iterator:
        """Stream the tool

        Raises:
            NotImplementedError: The result

        Yields:
            Iterator[typing.Iterator]: The result of the call
        """
        if is_async_function(self.option.f):
            raise NotImplementedError
        elif is_generator_function(self.option.f):
            for k in self.option.f(**self.args):
                yield k
        else:
            yield self.option.f(**self.args)
        
    async def astream(self):
        """Stream the tool

        Yields:
            Iterator[typing.Iterator]: The result of the call
        """
        if is_generator_function(self.option.f) and is_async_function(self.option.f):
            async for k in await self.option.f(**self.args):
                yield k
        elif is_generator_function(self.option.f):
            for k in await self.option.f(**self.args):
                yield k
        elif is_async_function(self.option.f):
            yield await self.option.f(**self.args)
        else:
            yield self.option.f(**self.args)


class ToolGen(object):
    """Object to keep track of adding tools
    """

    def __init__(self, tools: ToolSet, add_to: typing.List):
        """

        Args:
            tools (ToolSet): 
            add_to (typing.List): 
        """

        self.tools = tools
        self.add_to = add_to
        self.cur_tool = None
        self.continuing = False

    def add(self, name: str, args: str) -> ToolCall:
        """Add a new tool to the tool generator

        Args:
            name (str): 
            args (str): 

        Returns:
            ToolCall: 
        """
        tool_call = {
            'name': name,
            'args': args,
            'complete': False
        }
        if self.cur_tool is not None:
            self.cur_tool['completed'] = True

            tool = ToolCall(
                option=self.tools[self.cur_tool['name']], 
                args=json.loads(self.cur_tool['args'])
            )
        else:
            tool = None
        
        print('Add Tool: ', self.cur_tool)
        self.cur_tool = tool_call
        self.continuing = True
        return tool

    def append(self, args: str):
        """Add args 

        Args:
            args (str): Add the args

        Raises:
            RuntimeError: If there is no current tool
        """
        print('Append Tool: ', self.cur_tool)
        if self.cur_tool is None:
            raise RuntimeError('Cur tool is not specified but appending')
        self.cur_tool['args'] += args

    def end(self) -> ToolCall:
        """

        Returns:
            ToolCall: 
        """
        if self.cur_tool is None:
            self.continuing = False
            return None
        
        print(self.tools.tools, self.cur_tool['name'])
        tool = ToolCall(
            option=self.tools[self.cur_tool['name']], 
            args=json.loads(self.cur_tool['args'])
        )
        self.cur_tool = None
        self.continuing = False
        return tool


def exclude_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    """
    Filter messages by excluding specified roles.
    This function takes an iterable of messages and one or more role strings, returning
    a new list containing only messages whose roles are not in the specified roles to exclude.
    Args:
        messages (typing.Iterable[Msg]): An iterable of message objects
        *role (str): Variable number of role strings to exclude
    Returns:
        typing.List[Msg]: A list of messages excluding those with specified roles
    Example:
        >>> messages = [Msg(role="user", content="hi"), Msg(role="system", content="hello")]
        >>> exclude_role(messages, "system")
        [Msg(role="user", content="hi")]
    """
    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): 

    Returns:
        typing.List[Msg]: 
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]



def to_dialog(prompt: typing.Union[Dialog, Msg]) -> Dialog:
    """Convert a prompt to a dialog

    Args:
        prompt (typing.Union[Dialog, Msg]): The prompt to convert
    """
    if isinstance(prompt, Msg):
        prompt = ListDialog([prompt])

    return prompt


class Response(ABC):
    """
    """

    @abstractmethod
    def __call__(self, response, msg: Msg) -> typing.Tuple[Msg, typing.Any]:
        pass

    @abstractmethod
    def delta(self, response, msg: Msg) -> typing.Tuple[Msg, typing.Any]: 
        pass

    @abstractmethod
    def prep(self) -> typing.Dict:
        pass


def llm_forward(
    f: typing.Callable, 
    *args, 
    _respond: typing.List[Response]=None, 
    _role: str='assistant',
    **kwargs
):
    """
    Executes a given function with specified arguments and processes the response.
    Args:
        f (typing.Callable): The function to be executed.
        *args: Variable length argument list to be passed to the function.
        _respond (typing.List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to be passed to the function.
    Returns:
        tuple: A tuple containing the final message (Msg) and the last value processed by the Response objects.
    """
    _respond = _respond or []
    msg = Msg(
        role=_role
    )

    for r in _respond:
        kwargs.update(r.prep())

    response = f(
        *args, **kwargs
    )
    vals = []

    for r in _respond:
        msg, val = r(response, msg)
        vals.append(val)
    return msg, val


async def llm_aforward(
    f, 
    *args, 
    _respond: typing.List[Response]=None, 
    _role: str='assistant',
    **kwargs
):
    """
    Asynchronously forwards a function call with additional response processing.
    Args:
        f (Callable): The function to be called.
        *args: Positional arguments to pass to the function.
        _respond (List[Response], optional): A list of Response objects to process the function's response. Defaults to None.
        _role (str, optional): The role to be assigned to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        Tuple[Msg, Any]: A tuple containing the processed message and the final value from the response processing.
    """
    _respond = _respond or []
    msg = Msg(
        role=_role
    )

    for r in _respond:
        kwargs.update(r.prep())

    response = await f(
        *args, **kwargs
    )
    vals = []

    for r in _respond:
        msg, val = r(response, msg)
        vals.append(val)
    return msg, val


def llm_stream(
    f: typing.Callable, 
    *args, 
    _respond: typing.List[Response]=None, 
    _role: str='assistant',
    **kwargs
):
    """
    Streams responses from a language model function, allowing for intermediate processing.
    Args:
        f (typing.Callable): The language model function to call.
        *args: Positional arguments to pass to the language model function.
        _respond (typing.List[Response], optional): A list of Response objects for processing the model's output. Defaults to None.
        _role (str, optional): The role to assign to the message. Defaults to 'assistant'.
        **kwargs: Additional keyword arguments to pass to the language model function.
    Yields:
        Tuple[Msg, Any]: A tuple containing the message object and the processed value from the response.
    """
    _respond = _respond or []
    msg = Msg(
        role=_role
    )

    for r in _respond:
        kwargs.update(r.prep())

    for response in f(
        *args, **kwargs
    ):
        vals = []

        for r in _respond:
            msg, val = r(response, msg)
            vals.append(val)
        yield msg, val


async def llm_astream(
    f: typing.Callable, 
    *args, 
    _respond: typing.List[Response]=None, 
    _role: str='assistant',
    **kwargs
) -> typing.AsyncIterator:
    """_summary_

    Args:
        f (typing.Callable): The function to run
        _respond (typing.List[Response], optional): The processes to use for responding
        _role (str, optional): The role for . Defaults to 'assistant'.

    Returns:
        typing.AsyncIterator: 

    Yields:
        typing.AsyncIterator: The Message and the results
    """
    _respond = _respond or []
    msg = Msg(
        role=_role
    )

    for r in _respond:
        kwargs.update(r.prep())

    async for response in await f(
        *args, **kwargs
    ):
        vals = []

        for r in _respond:
            msg, val = r(response, msg)
            vals.append(val)
        yield msg, val


class ToMsg(ABC):
    """Converts the input to a message
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Msg:
        pass


class ToText(ToMsg):
    """Converts the input to a text message
    """

    def __init__(self, role: str, field: str='content'):
        """_summary_

        Args:
            role (str): The role for the message
            field (str, optional): The name of the field for the text. Defaults to 'content'.
        """
        self.role = role
        self.field = field

    def __call__(self, text: str) -> Msg:
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


# class ConvMsg(ABC):
#     """Converts inputs to a message"""

#     @abstractmethod
#     def from_msg(self, msg: Msg) -> typing.Any:
#         pass

#     @abstractmethod
#     def to_msg(self, *args, **kwds) -> Msg:
#         pass


# class ConvStr(ConvMsg):
#     """Converts the inputs to a standard message"""

#     def __init__(
#         self, 
#         role: str="system", 
#         text_name: str='content',
#     ):
#         super().__init__()
#         self.text_name = text_name
#         self.role = role

#     def from_msg(self, response: Msg) -> typing.Any:
#         """Convert the response from a messagew

#         Args:
#             msg (Msg): The message to convert

#         Returns:
#             typing.Any: The result
#         """
#         return response[self.text_name]

#     def to_msg(self, prompt: str) -> Msg:
#         """Convert the inputs to a message"""
#         kwargs = {
#             self.text_name: prompt
#         }
#         return Msg(role=self.role, **kwargs)

#     def from_delta(self, response: Msg) -> typing.Any:
#         """Convert the response from a messagew

#         Args:
#             msg (Msg): The message to convert

#         Returns:z
#             typing.Any: The result
#         """
#         return response.delta[self.text_name]



# class LLM(Module, ABC):
#     """APIAdapter allows one to adapt various WebAPI or otehr
#     API for a consistent interface
#     """

#     def user(
#         self, 
#         delta: typing.Dict=None, 
#         meta: typing.Dict=None, 
#         type_:str ='data', **kwargs
#     ) -> Msg:
#         """Create a user message

#         Args:
#             type_ (str, optional): The type of the message. Defaults to 'data'.
#             delta (typing.Dict, optional): The change in the message. Defaults to None.
#             meta (typing.Dict, optional): Any other details. Defaults to None.

#         Returns:
#             Msg: The message
#         """
#         return Msg(
#             role='user',
#             meta=meta, 
#             delta=delta, type_=type_, **kwargs
#         )

#     def assistant(
#         self, 
#         delta: typing.Dict=None, 
#         meta: typing.Dict=None, type_:str ='data', 
#         **kwargs
#     ) -> Msg:
#         """Create an assistant message

#         Args:
#             type_ (str, optional): The type of the message. Defaults to 'data'.
#             delta (typing.Dict, optional): The change in the message. Defaults to None.
#             meta (typing.Dict, optional): Any other details. Defaults to None.

#         Returns:
#             Msg: The message
#         """
#         return Msg(
#             role='assistant',
#             meta=meta, 
#             delta=delta, type_=type_, **kwargs
#         )

#     def system(
#         self, 
#         delta: typing.Dict=None, 
#         meta: typing.Dict=None, type_:str ='data', **kwargs
#     ) -> Msg:
#         """Create a system message

#         Args:
#             type_ (str, optional): The type of the message. Defaults to 'data'.
#             delta (typing.Dict, optional): The change in the message. Defaults to None.
#             meta (typing.Dict, optional): Any other details. Defaults to None.

#         Returns:
#             Msg: The message
#         """
#         return Msg(
#             role='system',
#             meta=meta, 
#             delta=delta, type_=type_, **kwargs
#         )

#     def tool_resp(
#         self, type_:str ='tool', 
#         delta: typing.Dict=None, 
#         meta: typing.Dict=None, **kwargs
#     ) -> Msg:
#         """Create a tool message

#         Args:
#             type_ (str, optional): The type of the message. Defaults to 'tool'.
#             delta (typing.Dict, optional): The change in the message. Defaults to None.
#             meta (typing.Dict, optional): Any other details. Defaults to None.

#         Returns:
#             Msg: The message
#         """
#         return Msg(
#             role='tool',
#             meta=meta, 
#             delta=delta, type_=type_, **kwargs
#         )

#     def msg(
#         self, role: str, *args,
#         type_: str='data', 
#         meta: typing.Dict=None, 
#         delta: typing.Dict=None, 
#         **kwargs
#     ) -> Msg:
#         """The message

#         Args:
#             type_ (str, optional): . Defaults to 'data'.
#             meta (typing.Dict, optional): . Defaults to None.
#             delta (typing.Dict, optional): . Defaults to None.

#         Returns:
#             Msg: 
#         """
#         try:
#             f = object.__getattribute__(self, role)
#         except AttributeError:
#             raise AttributeError(
#                 f'There is no role named {role}.'
#             )
#         return f(
#             *args, role=role,
#             type_=type_, meta=meta,
#             delta=delta, **kwargs
#         )

#     @abstractmethod
#     def forward(
#         self, prompt: LLM_PROMPT, 
#         **kwarg_override
#     ) -> typing.Tuple[Msg, typing.Any]:
#         """Run a standard query to the API

#         Args:
#             data : Data to pass to the API

#         Returns:
#             typing.Dict: The result of the API call
#         """
#         pass

#     def stream(
#         self, prompt: LLM_PROMPT, 
#         **kwarg_override
#     ) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
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
#         yield self.forward(prompt, **kwarg_override)
    
#     async def aforward(
#         self, prompt: LLM_PROMPT, 
#         **kwarg_override
#     ) -> LLM_RESPONSE:
#         """Run this query for asynchronous operations
#         The default behavior is simply to call the query

#         Args:
#             data: Data to pass to the API

#         Returns:
#             typing.Any: 
#         """
#         return self.forward(prompt, **kwarg_override)

#     async def astream(
#         self, prompt: LLM_PROMPT, **kwarg_override
#     ) -> typing.AsyncIterator[LLM_RESPONSE]:
#         """Run this query for asynchronous streaming operations
#         The default behavior is simply to call the query

#         Args:
#             prompt (AIPrompt): The data to pass to the API

#         Yields:
#             typing.Dict: The data returned from the API
#         """
#         result = self.forward(prompt, **kwarg_override)
#         yield result

#     def __call__(self, prompt: LLM_PROMPT, **kwarg_override) -> Msg:
#         """Execute the AIModel

#         Args:
#             prompt (AIPrompt): The prompt

#         Returns:
#             AIResponse: Get the response from the AI
#         """
#         return self.forward(prompt, **kwarg_override)
    
#     def get(self, x: typing.Union[str, typing.Callable], dx: typing.Union[str, typing.Callable]):
#         """
#         Retrieves a value based on the provided parameters.

#             x (typing.Union[str, typing.Callable]): The key or a callable to determine the key.
#             dx (typing.Union[str, typing.Callable]): The default value or a callable to determine the default value.

#             Get: An instance of the Get class initialized with the provided parameters.
#         """
#         return Get(self, x, dx)


# class Get(Module):
#     """Use to convert a message response to a value 
#     """
#     def __init__(self, llm: LLM, x: typing.Union[str, typing.Callable], dx: typing.Union[str, typing.Callable]):
#         """Initialize the AI class instance.

#             llm (LLM): Language Learning Model instance to be used for processing.
#             x (Union[str, Callable]): Input content or function that returns content to be processed.
#             dx (Union[str, Callable]): Delta or differential content/function that modifies or supplements the input.

#         """
#         super().__init__()
#         self._llm = llm
#         self._dx = dx
#         self._x = x

#     def forward(self, prompt, **kwarg_override) -> typing.Any:
#         """
#         Forward the prompt through the LLM and extract/process the response.
#         Args:
#             prompt: Input prompt to be processed by the LLM
#             **kwarg_override: Optional keyword arguments to override default LLM parameters
#         Returns:
#             typing.Any: Either a direct string extraction from the LLM response using self._x as a key,
#                        or the result of processing the response through self._x as a callable function.
#         Yields:
#             typing.Any: The processed response, either extracted or transformed based on self._x configuration
  
#         """
#         msg, _ = self._llm.forward(
#             prompt, **kwarg_override
#         )
#         return msg[self._x]
    
#     async def aforward(
#         self, prompt, **kwarg_override
#     ) -> typing.Any:
#         """
#         Asynchronously forwards a prompt to the language model and yields processed results.
#         This method sends a prompt to the underlying language model and processes its response
#         according to the configured extraction pattern (_x).
#         Parameters
#         ----------
#         prompt : Any
#             The input prompt to be sent to the language model.
#         **kwarg_override : dict
#             Optional keyword arguments to override default parameters of the language model.
#         Yields
#         ------
#         Any
#             The processed response from the language model. If _x is a string, yields the value
#             corresponding to that key from the response. If _x is a callable, yields the result
#             of applying that function to the response.
#         Returns
#         -------
#         list
#             A single-element list containing the extraction pattern (_x).
#         Notes
#         -----
#         The extraction pattern (_x) determines how the response is processed:
#         - If _x is a string: extracts the value at that key from the response dictionary
#         - If _x is a callable: applies the function to the entire response
#         """
#         msg, _ = self._llm.aforward(prompt, **kwarg_override)
#         return msg[self._x]
    
#     def stream(
#         self, prompt, **kwarg_override
#     ) -> typing.Iterator:
#         """ Stream the results from the LLM and extract/process the response.
#         Args:
#             prompt: Input prompt to be processed by the LLM
#             **kwarg_override: Optional keyword arguments to override default LLM parameters
#         Returns:
#             typing.Iterator: An iterator that yields the processed response, either extracted or transformed based on self._x configuration
#         Yields: 
#             typing.Any: The processed response, either extracted or transformed based on self._x configuration
#         """ 
#         for msg, _ in self._llm.stream(
#             prompt, **kwarg_override
#         ):
#             if isinstance(self._dx, str):
#                 yield msg[self._dx]
#             else:
#                 yield self._dx(msg)
    
#     async def astream(self, prompt, **kwarg_override) -> typing.AsyncIterator:
#         """Asynchronously stream the results from the LLM and extract/process the response.
#         Args:
#             prompt: Input prompt to be processed by the LLM
#             **kwarg_override: Optional keyword arguments to override default LLM parameters
#         Returns:
#             typing.AsyncIterator: An asynchronous iterator that yields the processed response, either extracted or transformed based on self._x configuration
#         Yields:
#             typing.Any: The processed response, either extracted or transformed based on self._x configuration
#         """
#         async for msg in self._llm.astream(prompt, **kwarg_override):
#             if isinstance(self._dx, str):
#                 yield msg[self._dx]
#             else:
#                 yield self._dx(msg)

