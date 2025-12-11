"""

This module defines various processing classes and functions for synchronous and asynchronous operations and the Process interfaces.

The module also includes utility functions for forwarding, streaming, mapping, and reducing operations across these processing classes.

The Process interfaces are:

class Process
- forward(self, *args, **kwargs) -> t.Any:
class AsyncProcess
- aforward(self, *args, **kwargs) -> t.Any:
class StreamProcess
- stream(self, *args, **kwargs) -> t.Iterator[t.Any]:
class AsyncStreamProcess
- astream(self, *args, **kwargs) -> t.AsyncIterator:t.Any

"""

# 1st party
from abc import ABC, abstractmethod
import numpy as np
import typing as t

# 3rd party
import pydantic

# local
from dachi.core import Module
from dachi.utils.func import (
    is_async_function,
    is_generator_function,
    is_async_generator_function,
    is_iterator,
    is_async_iterator

)
from ._arg_model import (
    Ref,
    BaseArgs,
    func_arg_model
)

S = t.TypeVar('S', bound=pydantic.BaseModel)


class Process(Module):
    """
    Base class for synchronous processing modules.
    It inherits from BaseModule and implements the forward method.

    Refer to the BaseModule documentation for details on field definitions and initialization.

    """
    ForwardArgModel: t.ClassVar = None
    ForwardRefArgModel: t.ClassVar = None
    ForwardProcessCall: t.ClassVar = None
    ForwardRefProcessCall: t.ClassVar = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is Process:
            return

        try:
            cls.ForwardArgModel = func_arg_model(cls, cls.forward)
            cls.ForwardRefArgModel = func_arg_model(cls, cls.forward, with_ref=True)
            cls.ForwardProcessCall = ProcessCall[cls, cls.ForwardArgModel]
            cls.ForwardRefProcessCall = ProcessCall[cls, cls.ForwardRefArgModel]
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to generate argument model for {cls.__name__}.forward: {e}\n"
                f"ProcessCall creation will not be available for this class.",
                UserWarning,
                stacklevel=2
            )
            cls.ForwardArgModel = None
            cls.ForwardRefArgModel = None
            cls.ForwardProcessCall = None
            cls.ForwardRefProcessCall = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        pass

    def __call__(self, *args, **kwargs) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        return self.forward(*args, **kwargs)
    
    def forward_process_call(
        self,
        _ref: bool=False,
        **kwargs,
    ) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        if _ref:
            if self.ForwardRefArgModel is None:
                raise RuntimeError(
                    f"Cannot create ProcessCall for {self.__class__.__name__}: "
                    "argument model generation failed during class initialization"
                )
            arg_model = self.ForwardRefArgModel(**kwargs)
            return self.ForwardRefProcessCall(process=self, args=arg_model)

        if self.ForwardArgModel is None:
            raise RuntimeError(
                f"Cannot create ProcessCall for {self.__class__.__name__}: "
                "argument model generation failed during class initialization"
            )
        arg_model = self.ForwardArgModel(**kwargs)
        return self.ForwardProcessCall(process=self, args=arg_model)


PROCESS = t.TypeVar('PROCESS', bound=Process)


class AsyncProcess(Module):
    """Base class for Async Processes. It defines the
    aforward method that must be implemented by subclasses.
    Refer to the BaseModule documentation for details on field definitions and initialization.

    """

    AForwardArgModel: t.ClassVar['BaseArgs'] = None
    AForwardRefArgModel: t.ClassVar['BaseArgs'] = None
    AForwardProcessCall: t.ClassVar['BaseArgs'] = None
    AForwardRefProcessCall: t.ClassVar['BaseArgs'] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is AsyncProcess:
            return

        try:
            cls.AForwardArgModel = func_arg_model(cls, cls.aforward)
            cls.AForwardRefArgModel = func_arg_model(cls, cls.aforward, with_ref=True)
            cls.AForwardProcessCall = AsyncProcessCall[cls, cls.AForwardArgModel]
            cls.AForwardRefProcessCall = AsyncProcessCall[cls, cls.AForwardRefArgModel]
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to generate argument model for {cls.__name__}.aforward: {e}\n"
                f"AsyncProcessCall creation will not be available for this class.",
                UserWarning,
                stacklevel=2
            )
            cls.AForwardArgModel = None
            cls.AForwardRefArgModel = None
            cls.AForwardProcessCall = None
            cls.AForwardRefProcessCall = None

    @abstractmethod
    async def aforward(
        self, 
        *args, 
        **kwargs
    ) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        pass

    def aforward_process_call(
        self,
        _ref: bool=False,
        **kwargs,
    ) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        if _ref:
            if self.AForwardRefArgModel is None:
                raise RuntimeError(
                    f"Cannot create AsyncProcessCall for {self.__class__.__name__}: "
                    "argument model generation failed during class initialization"
                )
            arg_model = self.AForwardRefArgModel(**kwargs)
            return self.AForwardRefProcessCall(process=self, args=arg_model)

        if self.AForwardArgModel is None:
            raise RuntimeError(
                f"Cannot create AsyncProcessCall for {self.__class__.__name__}: "
                "argument model generation failed during class initialization"
            )
        arg_model = self.AForwardArgModel(**kwargs)
        return self.AForwardProcessCall(process=self, args=arg_model)


ASYNC_PROCESS = t.TypeVar('ASYNC_PROCESS', bound=AsyncProcess)

AP = t.TypeVar('AP', bound=AsyncProcess | Process)
ARGS = t.TypeVar('ARGS', bound=BaseArgs)


class BaseProcessCall(Module, t.Generic[ARGS]):

    args: ARGS

    def depends_on(self) -> t.Iterator[str]:
        """Get the names of processes this process depends on

        Yields:
            Iterator[str]: The names of the processes
        """
        for field_name in self.args.model_fields.keys():
            value = getattr(self.args, field_name)
            if isinstance(value, Ref):
                yield value.name


class ProcessCall(
    BaseProcessCall[ARGS], t.Generic[PROCESS, ARGS]
):
    process: PROCESS

    async def forward(self, **kwargs) -> t.Any:
        """Execute the wrapped process asynchronously

        Returns:
            t.Any: The output of the process
        """
        return await self.process.forward(**{
            k: (kwargs[v.data] if isinstance(v, Ref) else v)
            for k, v in self.args.model_dump().items()
        })


class AsyncProcessCall(
    BaseProcessCall[ARGS], t.Generic[ASYNC_PROCESS, ARGS]
):
    """Wrapper for a Process/AsyncProcess with its arguments in a DAG.

    Used by DataFlow to store both the process and its arguments together
    as a serializable unit. The name is stored as the key in DataFlow's
    processes ModuleDict.

    Args:
        process: The Process or AsyncProcess to execute
        args: Arguments to pass to the process (can be Ref or literal values)

    Note:
        ProcessCall is a data container, not an executable process. DataFlow
        extracts the process and args to execute them.

    Convenience Methods:
        is_async: Returns True if the wrapped process is AsyncProcess
    """
    process: ASYNC_PROCESS

    async def aforward(self, **kwargs) -> t.Any:
        """Execute the wrapped process asynchronously

        Returns:
            t.Any: The output of the process
        """
        return await self.process.aforward(**{
            k: (kwargs[v.data] if isinstance(v, Ref) else v)
            for k, v in self.args.model_dump().items()
        })


class StreamProcess(Module):
    """Base class for Stream Processes. It defines the
    stream method that must be implemented by subclasses.

    Refer to the BaseModule documentation for details on field definitions and initialization.
    """

    StreamArgModel: t.ClassVar = None
    StreamRefArgModel: t.ClassVar = None
    StreamProcessCall: t.ClassVar = None
    StreamRefProcessCall: t.ClassVar = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is StreamProcess:
            return

        try:
            cls.StreamArgModel = func_arg_model(cls, cls.stream)
            cls.StreamRefArgModel = func_arg_model(cls, cls.stream, with_ref=True)
            cls.StreamProcessCall = StreamProcessCall[cls, cls.StreamArgModel]
            cls.StreamRefProcessCall = StreamProcessCall[cls, cls.StreamRefArgModel]
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to generate argument model for {cls.__name__}.stream: {e}\n"
                f"StreamProcessCall creation will not be available for this class.",
                UserWarning,
                stacklevel=2
            )
            cls.StreamArgModel = None
            cls.StreamRefArgModel = None
            cls.StreamProcessCall = None
            cls.StreamRefProcessCall = None

    @abstractmethod
    def stream(self, *args, **kwargs) -> t.Iterator[t.Any]:
        """Stream the output

        Yields:
            Iterator[t.Any]: The value streamed
        """
        pass

    def stream_process_call(
        self,
        _ref: bool=False,
        **kwargs,
    ) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        if _ref:
            if self.StreamRefArgModel is None:
                raise RuntimeError(
                    f"Cannot create StreamProcessCall for {self.__class__.__name__}: "
                    "argument model generation failed during class initialization"
                )
            arg_model = self.StreamRefArgModel(**kwargs)
            return self.StreamRefProcessCall(process=self, args=arg_model)

        if self.StreamArgModel is None:
            raise RuntimeError(
                f"Cannot create StreamProcessCall for {self.__class__.__name__}: "
                "argument model generation failed during class initialization"
            )
        arg_model = self.StreamArgModel(**kwargs)
        return self.StreamProcessCall(process=self, args=arg_model)


STREAM = t.TypeVar('STREAM', bound=StreamProcess)


class AsyncStreamProcess(Module):
    """Base class for AsyncStream Processes. It defines the
    stream method that must be implemented by subclasses.

    Refer to BaseModule documentation for details on field definitions and initialization.
    """

    AStreamArgModel: t.ClassVar = None
    AStreamRefArgModel: t.ClassVar = None
    AStreamProcessCall: t.ClassVar = None
    AStreamRefProcessCall: t.ClassVar = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is AsyncStreamProcess:
            return

        try:
            cls.AStreamArgModel = func_arg_model(cls, cls.astream)
            cls.AStreamRefArgModel = func_arg_model(cls, cls.astream, with_ref=True)
            cls.AStreamProcessCall = AsyncStreamProcessCall[cls, cls.AStreamArgModel]
            cls.AStreamRefProcessCall = AsyncStreamProcessCall[cls, cls.AStreamRefArgModel]
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to generate argument model for {cls.__name__}.astream: {e}\n"
                f"AsyncStreamProcessCall creation will not be available for this class.",
                UserWarning,
                stacklevel=2
            )
            cls.AStreamArgModel = None
            cls.AStreamRefArgModel = None
            cls.AStreamProcessCall = None
            cls.AStreamRefProcessCall = None

    @abstractmethod
    async def astream(self, *args, **kwargs) -> t.AsyncIterator:
        """
        Returns:
            Streamer: The Streamer to loop over
        """
        pass

    def astream_process_call(
        self,
        _ref: bool=False,
        **kwargs,
    ) -> t.Any:
        """Execute the module

        Returns:
            t.Any: The output of the module
        """
        if _ref:
            if self.AStreamRefArgModel is None:
                raise RuntimeError(
                    f"Cannot create AsyncStreamProcessCall for {self.__class__.__name__}: "
                    "argument model generation failed during class initialization"
                )
            arg_model = self.AStreamRefArgModel(**kwargs)
            return self.AStreamRefProcessCall(process=self, args=arg_model)

        if self.AStreamArgModel is None:
            raise RuntimeError(
                f"Cannot create AsyncStreamProcessCall for {self.__class__.__name__}: "
                "argument model generation failed during class initialization"
            )
        arg_model = self.AStreamArgModel(**kwargs)
        return self.AStreamProcessCall(process=self, args=arg_model)


ASYNC_STREAM = t.TypeVar('ASYNC_STREAM', bound=AsyncStreamProcess)



class StreamProcessCall(
    BaseProcessCall[ARGS], t.Generic[STREAM, ARGS]
):
    process: STREAM

    def stream(self, **kwargs) -> t.Iterator[t.Any]:
        """Execute the wrapped process asynchronously

        Returns:
            t.Any: The output of the process
        """
        return self.process.stream(**{
            k: (kwargs[v.data] if isinstance(v, Ref) else v)
            for k, v in self.args.model_dump().items()
        })


class AsyncStreamProcessCall(
    BaseProcessCall[ARGS], t.Generic[ASYNC_STREAM, ARGS]
):
    process: ASYNC_STREAM

    async def astream(self, **kwargs) -> t.AsyncIterator[t.Any]:
        """Execute the wrapped process asynchronously

        Returns:
            t.Any: The output of the process
        """
        return await self.process.astream(**{
            k: (kwargs[v.data] if isinstance(v, Ref) else v)
            for k, v in self.args.model_dump().items()
        })


def forward(
    f: t.Union[Process, t.Callable], 
    *args, **kwargs
) -> t.Any:
    """
    Calls the forward method on the module or the function that has been passed in.
    Parameters:
    f (t.Union[Module, t.Callable]): The module or function to forward to.
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.
    Returns:
    t.Any: The result of the forward call.
    Raises:
    NotImplementedError: If the function is asynchronous.
    RuntimeError: If the function type is not supported.
    """ 
    if isinstance(f, Process):
        return f.forward(*args, **kwargs)
    if not is_async_function(f) and not is_generator_function(f):
        return f(*args, **kwargs)
    if not is_async_function(f) and is_generator_function(f):
        return [v for v in f(*args, **kwargs)]
    if is_async_function(f) and not is_generator_function(f):
        raise NotImplementedError('Cannot forward with async function')
    raise RuntimeError()


async def aforward(
    f: t.Union[Process, t.Callable], 
    *args, **kwargs
) -> t.Any:
    """
    Asynchronously calls the appropriate forward method or function.
    This function determines the type of the input `f` and calls the corresponding
    forward method or function, handling both synchronous and asynchronous cases,
    as well as generator functions.
    Parameters:
    f (t.Union[Module, t.Callable]): The module or callable to be executed.
    *args: Variable length argument list to be passed to the callable.
    **kwargs: Arbitrary keyword arguments to be passed to the callable.
    Returns:
    t.Any: The result of the forward method or function call, which can be
    synchronous or asynchronous, and can handle generator functions.
    """
    
    if isinstance(f, AsyncProcess):
        return await f.aforward(*args, **kwargs)
    if isinstance(f, Process):
        return f.forward(*args, **kwargs)
    if not is_async_function(f) and not is_generator_function(f):
        if not isinstance(f, t.Callable):
            raise TypeError(
                f"Object {object} is not callable"
            )
        return f(*args, **kwargs)
    if is_async_function(f) and not is_generator_function(f):
        if not isinstance(f, t.Callable):
            raise TypeError(
                f"Object {object} is not callable"
            )
        return await f(*args, **kwargs)
    # if not is_async_function(f) and is_generator_function(f):
    #     return [v for v in f(*args, **kwargs)]
    # if is_async_function(f) and is_generator_function(f):
    #     return [v async for v in await f(*args, **kwargs)]
    raise RuntimeError(
        f"Cannot execute forward with {f}"
    )


def stream(f: t.Union[StreamProcess, t.Callable], *args, **kwargs) -> t.Any:
    """
    Stream values from a given function or StreamModule.
    This function handles different types of input functions or modules and streams their output.
    It supports synchronous generator functions and StreamModules. It raises exceptions for
    unsupported async functions or async generator functions.
    Args:
        f (t.Union[Module, t.Callable]): The function or StreamModule to stream from.
        *args: Positional arguments to pass to the function or StreamModule.
        **kwargs: Keyword arguments to pass to the function or StreamModule.
    Yields:
        t.Any: The values yielded by the function or StreamModule.
    Raises:
        NotImplementedError: If an async function or async generator function is passed.
        RuntimeError: If the input does not match any supported type.
    """
    
    if isinstance(f, StreamProcess):
        for v in f.stream(*args, **kwargs):
            yield v
    elif (is_async_function(f) and is_generator_function(f)) or is_async_generator_function(f):
        raise TypeError(
            'Cannot execute an async streaming function from a streaming function')
    elif not is_async_function(f) and is_generator_function(f):
        for v in f(*args, **kwargs):
            yield v
    elif is_async_function(f) and not is_generator_function(f):
        raise TypeError('Cannot stream with async function')
    elif not is_async_function(f) and not is_generator_function(f):
        res = f(*args, **kwargs)
        yield res
    else:
        raise TypeError()


async def astream(f: t.Union[AsyncStreamProcess, t.Callable], *args, **kwargs) -> t.Any:
    """
    Stream values from a given function or AsyncStreamModule.
    This function handles different types of input functions or modules and streams their output.
    It supports synchronous generator functions and StreamModules. It raises exceptions for
    unsupported async functions or async generator functions.
    Args:
        f (t.Union[Module, t.Callable]): The function or StreamModule to stream from.
        *args: Positional arguments to pass to the function or StreamModule.
        **kwargs: Keyword arguments to pass to the function or StreamModule.
    Yields:
        t.Any: The values yielded by the function or StreamModule.
    Raises:
        NotImplementedError: If an async function or async generator function is passed.
        RuntimeError: If the input does not match any supported type.
    """
    if isinstance(f, AsyncStreamProcess):
        async for v in await f.astream(*args, **kwargs):
            yield v

    elif isinstance(f, StreamProcess):
        for v in f.stream(*args, **kwargs):
            yield v
    elif (is_async_function(f) and is_generator_function(f)) or is_async_generator_function(f):
        async for v in f(*args, **kwargs):
            yield v
    elif is_generator_function(f):

        for v in f(*args, **kwargs):
            yield v

    elif is_iterator(f):
        for v in f:
            yield v

    elif is_async_iterator(f):
        async for v in f:
            yield v

    elif is_async_function(f):
        yield await f(*args, **kwargs)
    else:
        yield f(*args, **kwargs)




PA = t.TypeVar('PA', bound=Process | AsyncProcess)




class Func(Process):
    """Function process that applies a callable to the input data.
    """
    f: t.Callable
    args: t.List[t.Any] = pydantic.Field(default_factory=list)
    kwargs: t.Dict[str, t.Any] = pydantic.Field(default_factory=dict)

    def forward(self, *args, **kwargs):
        """
        Forward the input data through the function process.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        return self.f(
            *self.args, *args, **self.kwargs, **kwargs
        )


class AsyncFunc(AsyncProcess):
    """A function wrapper
    """
    f: t.Callable
    args: t.List[t.Any] = pydantic.Field(default_factory=list)
    kwargs: t.Dict[str, t.Any] = pydantic.Field(default_factory=dict)

    async def aforward(self, *args, **kwargs):

        return await self.f(
            *self.args, *args, **self.kwargs, **kwargs
        )
