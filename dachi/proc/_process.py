"""

This module defines various processing classes and functions for synchronous and asynchronous operations and the Process interfaces.

The module also includes utility functions for forwarding, streaming, mapping, and reducing operations across these processing classes.

The Process interfaces are:

class Process
- forward(self, *args, **kwargs) -> typing.Any:
class AsyncProcess
- aforward(self, *args, **kwargs) -> typing.Any:
class StreamProcess
- stream(self, *args, **kwargs) -> typing.Iterator[typing.Any]:
class AsyncStreamProcess
- astream(self, *args, **kwargs) -> typing.AsyncIterator:typing.Any

"""

# 1st party
import inspect
from abc import ABC, abstractmethod
from functools import partial
import typing
from typing import Any
import asyncio
import itertools
import dataclasses
import numpy as np
import typing as t

# 3rd party
import pydantic

# local
from ..core import Module, ModuleList
from ..utils import (
    is_async_function,
    is_generator_function,
    is_async_generator_function,
    is_iterator,
    is_async_iterator

)

S = t.TypeVar('S', bound=pydantic.BaseModel)

@dataclasses.dataclass
class Ref:
    """Reference to the output of another process"""
    name: str


def func_arg_model(cls: type, cls_f, with_ref: bool=False) -> type[pydantic.BaseModel]:
    """
    Inspect `process_cls` and its method and build a Pydantic model for its args.
    Only handles keyword-style params; *args/**kwargs are ignored or forbidden.
    """
    sig = inspect.signature(cls_f)
    hints = t.get_type_hints(cls_f)

    fields: dict[str, tuple[t.Any, t.Any]] = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"{cls.__name__}.forward uses *args/**kwargs; "
                "cannot derive a static args model."
            )

        anno = hints.get(name, t.Any)

        if with_ref:
            anno = t.Union[Ref, anno]

        if param.default is inspect._empty:
            fields[name] = (anno, ...)
        else:
            fields[name] = (anno, param.default)

    model_name = f"{cls.__name__}Args"
    return create_model(model_name, **fields)  # type: ignore[call-arg]


class Process(Module, ABC):
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

        # Build an ArgsModel from `forward`
        cls.ForwardArgModel = func_arg_model(cls, cls.forward)
        cls.ForwardRefArgModel = func_arg_model(cls, cls.forward, with_ref=True)
        cls.ForwardProcessCall = ProcessCall[cls, cls.ForwardArgModel]
        cls.ForwardRefProcessCall = ProcessCall[cls, cls.ForwardRefArgModel]

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        return self.forward(*args, **kwargs)


PROCESS = typing.TypeVar('PROCESS', bound=Process)


class AsyncProcess(Module, ABC):
    """Base class for Async Processes. It defines the
    aforward method that must be implemented by subclasses.
    Refer to the BaseModule documentation for details on field definitions and initialization.

    """

    AForwardArgModel: t.ClassVar = None
    AForwardRefArgModel: t.ClassVar = None
    AForwardProcessCall: t.ClassVar = None
    AForwardRefProcessCall: t.ClassVar = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is Process:
            return

        # Build an ArgsModel from `forward`
        cls.AForwardArgModel = func_arg_model(cls, cls.aforward)
        cls.AForwardRefArgModel = func_arg_model(cls, cls.aforward, with_ref=True)
        cls.AForwardProcessCall = AsyncProcessCall[cls, cls.AForwardArgModel]
        cls.AForwardRefProcessCall = AsyncProcessCall[cls, cls.AForwardRefArgModel]

    @abstractmethod
    async def aforward(
        self, 
        *args, 
        **kwargs
    ) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        pass


ASYNC_PROCESS = typing.TypeVar('ASYNC_PROCESS', bound=AsyncProcess)


AP = typing.TypeVar('AP', bound=AsyncProcess | Process)
ARGS = typing.TypeVar('ARGS', bound=pydantic.BaseModel)


class ProcessCall(
    Module, typing.Generic[PROCESS, ARGS]
):
    process: PROCESS
    args: ARGS


class AsyncProcessCall(
    Module, typing.Generic[ASYNC_PROCESS, ARGS]
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
    process: AsyncProcess
    args: ARGS


class StreamProcess(Module, ABC):
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

        if cls is Process:
            return

        # Build an ArgsModel from `forward`
        cls.StreamArgModel = func_arg_model(cls, cls.stream)
        cls.StreamRefArgModel = func_arg_model(cls, cls.stream, with_ref=True)
        cls.StreamProcessCall = StreamProcessCall[cls, cls.StreamArgModel]
        cls.StreamRefProcessCall = StreamProcessCall[cls, cls.StreamRefArgModel]

    @abstractmethod
    def stream(self, *args, **kwargs) -> typing.Iterator[typing.Any]:
        """Stream the output

        Yields:
            Iterator[typing.Any]: The value streamed
        """
        pass


STREAM = typing.TypeVar('STREAM', bound=StreamProcess)


class AsyncStreamProcess(Module, ABC):
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

        if cls is Process:
            return

        # Build an ArgsModel from `forward`
        cls.AStreamArgModel = func_arg_model(cls, cls.astream)
        cls.AStreamRefArgModel = func_arg_model(cls, cls.astream, with_ref=True)
        cls.AStreamProcessCall = AsyncStreamProcessCall[cls, cls.AStreamArgModel]
        cls.AStreamRefProcessCall = AsyncStreamProcessCall[cls, cls.AStreamRefArgModel]

    @abstractmethod
    async def astream(self, *args, **kwargs) -> typing.AsyncIterator:
        """
        Returns:
            Streamer: The Streamer to loop over
        """
        pass


ASYNC_STREAM = typing.TypeVar('ASYNC_STREAM', bound=AsyncStreamProcess)



class StreamProcessCall(
    StreamProcess, typing.Generic[STREAM, ARGS]
):
    process: STREAM
    args: ARGS



class AsyncStreamProcessCall(
    AsyncStreamProcess, typing.Generic[ASYNC_STREAM, ARGS]
):
    process: ASYNC_STREAM
    args: ARGS



def forward(
    f: typing.Union[Process, typing.Callable], 
    *args, **kwargs
) -> typing.Any:
    """
    Calls the forward method on the module or the function that has been passed in.
    Parameters:
    f (typing.Union[Module, typing.Callable]): The module or function to forward to.
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.
    Returns:
    typing.Any: The result of the forward call.
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
    f: typing.Union[Process, typing.Callable], 
    *args, **kwargs
) -> typing.Any:
    """
    Asynchronously calls the appropriate forward method or function.
    This function determines the type of the input `f` and calls the corresponding
    forward method or function, handling both synchronous and asynchronous cases,
    as well as generator functions.
    Parameters:
    f (typing.Union[Module, typing.Callable]): The module or callable to be executed.
    *args: Variable length argument list to be passed to the callable.
    **kwargs: Arbitrary keyword arguments to be passed to the callable.
    Returns:
    typing.Any: The result of the forward method or function call, which can be
    synchronous or asynchronous, and can handle generator functions.
    """
    
    if isinstance(f, AsyncProcess):
        return await f.aforward(*args, **kwargs)
    if isinstance(f, Process):
        return f.forward(*args, **kwargs)
    if not is_async_function(f) and not is_generator_function(f):
        if not isinstance(f, typing.Callable):
            raise TypeError(
                f"Object {object} is not callable"
            )
        return f(*args, **kwargs)
    if is_async_function(f) and not is_generator_function(f):
        if not isinstance(f, typing.Callable):
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


def stream(f: typing.Union[StreamProcess, typing.Callable], *args, **kwargs) -> typing.Any:
    """
    Stream values from a given function or StreamModule.
    This function handles different types of input functions or modules and streams their output.
    It supports synchronous generator functions and StreamModules. It raises exceptions for
    unsupported async functions or async generator functions.
    Args:
        f (typing.Union[Module, typing.Callable]): The function or StreamModule to stream from.
        *args: Positional arguments to pass to the function or StreamModule.
        **kwargs: Keyword arguments to pass to the function or StreamModule.
    Yields:
        typing.Any: The values yielded by the function or StreamModule.
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


async def astream(f: typing.Union[AsyncStreamProcess, typing.Callable], *args, **kwargs) -> typing.Any:
    """
    Stream values from a given function or AsyncStreamModule.
    This function handles different types of input functions or modules and streams their output.
    It supports synchronous generator functions and StreamModules. It raises exceptions for
    unsupported async functions or async generator functions.
    Args:
        f (typing.Union[Module, typing.Callable]): The function or StreamModule to stream from.
        *args: Positional arguments to pass to the function or StreamModule.
        **kwargs: Keyword arguments to pass to the function or StreamModule.
    Yields:
        typing.Any: The values yielded by the function or StreamModule.
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


# TDOO: Doesn't serialize
class Recur(pydantic.BaseModel):
    """Use to mark data that should not be batched when calling the map functions

    Example:
    recur = Recur(data="hello", n=5)
    for r in recur:
        print(r)
        # prints "hello" 5 times
    """
    data: typing.Any
    n: int

    @pydantic.field_validator('n')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('n must be greater than 0')
        return v

    def __iter__(self) -> typing.Iterator[typing.Any]:
        """Iterate over the object (n times)

        Returns:
            typing.Iterator[typing.Any]: The 
        """
        for _ in range(self.n):
            yield self.data


class Chunk(pydantic.BaseModel):
    """Use to mark data for batching
    As opposed to Recur, Chunk will batch the data into n chunks

    Example:
    chunk = Chunk(data=[1,2,3,4,5], n=2)
    for c in chunk:
        print(c)
        # prints [1,2] then [3,4,5]
    """
    data: typing.List
    n: int | None = None
    shuffle: bool = False
    use_last: bool = True

    def __iter__(self) -> typing.Iterator[typing.List]:
        """

        Yields:
            typing.Any: Get each value in the  
        """
        n = self.n
        data = np.array(self.data)
        count = n if n is not None else len(data)
        if self.shuffle:
            order = np.random.permutation(len(data))
        else:
            order = np.arange(0, len(data))
        
        chunk_size = len(data) // count
        cur = 0
        for d_i in range(count):
            if d_i == (count - 1) and self.use_last:
                if n is None:
                    yield data[order[cur]]
                else:
                    yield data[order[cur:]].tolist()
            else:
                if n is None:
                    yield data[order[cur]]
                else:
                    yield data[order[cur:cur+chunk_size]].tolist()
            cur = cur + chunk_size

    @property
    def sz(self) -> int:
        """Get the size of the chunk

        Returns:
            int: The size of the chunk
        """
        if self.n is not None:
            return self.n
        return len(list(self.data))

    @classmethod
    def m(
        cls, *data: typing.Iterable, n: int=None
    ) -> typing.Tuple['Chunk']:
        """Wrap multiple data through Ps

        data: The data to wrap in P
        n: The number of batches to have

        Returns:
            typing.Tuple[P]: The resulting ps
        """
        return tuple(
            Chunk(data=d, n=n) for d in data
        )


def chunk(
    *data: typing.List, n: int=None
) -> typing.Tuple[Chunk] | Chunk:
    """Wrap multiple data through Ps
    data: The data to wrap in P
    n: The number of batches to have
    Returns:
        typing.Tuple[Chunk]: The resulting ps
    """
    if len(data) == 1:
        return Chunk(data=data[0], n=n)
    return tuple(
        Chunk(data=d, n=n) for d in data
    )


def recur(
    *data: typing.Any, n: int
) -> typing.Tuple[Recur] | Recur:
    """Wrap multiple data through Ps
    data: The data to wrap in P
    n: The number of iterations to have
    Returns:
        typing.Tuple[Recur]: The resulting ps
    """
    if len(data) == 1:
        return Recur(data=data[0], n=n)
    return tuple(
        Recur(data=d, n=n) for d in data
    )

PA = typing.TypeVar('PA', bound=Process | AsyncProcess)


class Sequential(ModuleList[PA], Process, AsyncProcess):
    """
    Sequential class wraps multiple modules into a sequential list of modules that will be executed one after the other.
    Methods:
        __init__(*module) -> None:
            Initialize the Sequential with a list of modules.
            The modules can be any type of Process or AsyncProcess.
        add(module):
            Add a module to the Sequential.
        forward(*x) -> Any:
            Pass the input through each of the modules in sequence and return the result of the final module.
    """
    def forward(self, *x) -> Any:
        """Pass the input (x) through each of the modules

        Returns:
            Any: The result of the final model
        """
        multi = len(x) > 1
        if len(x) == 1:
            x = x[0]

        first = True
        for module in self:
            if first and multi:
                x = module(*x)
            else:
                x = module(x)
            first = False
        return x
    
    async def aforward(self, *x):
        """Asynchronously pass the input (x) through each of the modules

        Returns:
            Any: The result of the final model
        """
        multi = len(x) > 1
        if len(x) == 1:
            x = x[0]

        first = True
        for module in self:
            if first and multi:
                if isinstance(module, AsyncProcess):
                    x = await module.aforward(*x)
                else: # isinstance(module, Process):
                    x = module.forward(*x)
            else:
                if isinstance(module, AsyncProcess):
                    x = await module.aforward(x)
                else: # isinstance(module, Process):
                    x = module.forward(x)

            first = False
        return x


class AsyncParallel(
    ModuleList[ASYNC_PROCESS], AsyncProcess
):
    """Calls multiple modules in parallel and then 
    combines the results into a list

    Example:
    process = AsyncParallel(
        items=[process1, process2, process3]
    )
    y1, y2, y3 = process.aforward(x)
    """
    async def aforward(self, *args, **kwargs) -> typing.List:
        """The asynchronous method to use for inputs

        Args:
            args (typing.List[Args]): The list of inputs to the modules 

        Returns:
            typing.Tuple: The output to the paralell module
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for module, cur_args, cur_kwargs in process_loop(
                self.module_list, *args, **kwargs
            ):
                if isinstance(module, AsyncProcess):
                    tasks.append(tg.create_task(
                        module.aforward(*cur_args, **cur_kwargs)
                    ))
                elif isinstance(module, Process):
                    tasks.append(tg.create_task(
                        asyncio.to_thread(
                            module.forward, *cur_args, **cur_kwargs
                        )
                    ))

        return list(
            t.result() for t in tasks
        )


def process_loop(
    processes: typing.List | None, *args, **kwargs
) -> typing.Iterator[
    typing.Tuple[Process, typing.List, typing.Dict]
]:
    """Use to loop over the module list

    Args:
        modules (typing.Union[ModuleList, Module, None]): 

    Returns:
        typing.Iterator[ typing.Tuple[Module, typing.List, typing.Dict] ]: 

    Yields:
        Iterator[typing.Iterator[ typing.Tuple[Module, typing.List, typing.Dict] ]]: 
    """
    # if isinstance(processes, typing.List):
    #     processes = ModuleList(items=processes)

    sz = None
    parallizable = False
    if isinstance(processes, list):

        sz = len(processes)
        parallizable = True
    
    for arg in itertools.chain(args, kwargs.values()):
        print(arg)
        if isinstance(arg, Chunk):
            parallizable = True
            if sz is None:
                sz = arg.sz
            elif arg.sz != sz:
                raise ValueError(
                    'All inputs must have the same number '
                    'of items to parallelize'
                )
            print(2, sz)
        elif isinstance(arg, Recur):
            parallizable = True
    
    if not parallizable:
        raise ValueError('None of the inputs can be parallelized')
    
    if not isinstance(processes, list):
        processes = Recur(data=processes, n=sz)

    kwarg_names = list(kwargs.keys())
    kwarg_vals = [
        Recur(data=v, n=sz) 
        if not isinstance(v, Chunk) 
        else v for v in kwargs.values()
    ]
    args = [
        Recur(data=v, n=sz) 
        if not isinstance(v, Chunk) else v 
        for v in args
    ]
    
    for vs in zip(processes, *args, *kwarg_vals):
        m = vs[0]
        args = vs[1:len(args) + 1]
        kwargs = vs[len(args) + 1:]
        yield m, args, dict(zip(kwarg_names, kwargs))


def create_proc_task(
    tg: asyncio.TaskGroup,
    f: AsyncProcess | Process | typing.Callable,
    *args, 
    **kwargs
) -> typing.Any:
    """Create a task for the process to run in the task group

    This function creates a task in the provided asyncio TaskGroup for the given function `f`.
    It handles both synchronous and asynchronous functions, as well as Process and AsyncProcess types.
    
    Args:
        tg (asyncio.TaskGroup): The task group to create the task in
        f (AsyncProcess | Process | typing.Callable): The function to run
        *args: The arguments to pass to the function
        **kwargs: The keyword arguments to pass to the function
        
    Returns:

        typing.Any: The task created
    """

    if isinstance(f, AsyncProcess):
        
        return tg.create_task(f.aforward(*args, **kwargs))
    elif isinstance(f, Process):
        return tg.create_task(
            asyncio.to_thread(partial(f.forward, *args, **kwargs)))
    elif is_async_function(f):
        return tg.create_task(f(*args, **kwargs))
    return tg.create_task(asyncio.to_thread(partial(f, *args, *kwargs)))


def process_map(
    f: Process | AsyncProcess | typing.Callable, 
    *args, **kwargs
) -> typing.Tuple[typing.Any]:
    """Helper function to run async_map

    Args:
        f (Module): The function to asynchronously execute

    Returns:
        typing.Tuple[typing.Any]: The result of the map
    """
    return tuple(
        cur_f(*a, **kv) for cur_f, a, kv in process_loop(f, *args, **kwargs)
    )


async def async_process_map(
    f: AsyncProcess | typing.Callable, 
    *args, **kwargs
) -> typing.Tuple[typing.Any]:
    """Helper function to run async_map

    Args:
        f (Module): The function to asynchronously execute

    Returns:
        typing.Tuple[typing.Any]: The result of the map
    """
    tasks = []

    async with asyncio.TaskGroup() as tg:
        
        for cur_f, a, kv in process_loop(f, *args, **kwargs):
            tasks.append(
                create_proc_task(tg, cur_f, *a, **kv)
            )

    return tuple(
        t.result() for t in tasks
    )


def multiprocess(
    *f: Process | typing.Callable
) -> typing.Tuple[typing.Any]:
    """Helper function to run asynchronous processes

    Returns:
        typing.Tuple[typing.Any]: 
    """
    return tuple(
        f_i() for f_i in f
    )


async def async_multiprocess(
    *f: AsyncProcess | typing.Callable
) -> typing.Tuple[typing.Any]:
    """Helper function to run asynchronous processes

    Returns:
        typing.Tuple[typing.Any]: 
    """
    tasks = []
    async with asyncio.TaskGroup() as tg:

        for f_i in f:
            if isinstance(f_i, AsyncProcess):
                tasks.append(tg.create_task(f_i.aforward))
            else:
                tasks.append(tg.create_task(f_i))

    return tuple(
        task.result() for task in tasks
    )


# TODO: Update to be more flexible
def reduce(
    mod: Process, *args, init_mod: Process=None, init_val=None, **kwargs
) -> typing.Any:
    """Reduce the args passed in with a module

    Args:
        mod (Module): The module to use for reduction
        init_mod (Module, optional): The module to use for the first set of data. Defaults to None.
        init_val: The initial value to use
        
    Returns:
        The result of the reduction
    """
    results = []

    if init_val is not None:
        results.append(init_val)
    
    for _, cur_args, cur_kwargs in process_loop(None, *args, **kwargs):

        if len(results) == 0 and init_mod is not None and init_val is None:
            results.append(init_mod(*cur_args, **cur_kwargs))
        elif len(results) == 0 and init_mod is not None:
            results.append(init_mod(init_val, *cur_args, **cur_kwargs))
        else:
            if len(results) == 0:
                raise ValueError(
                    'No initial value or initial module provided. '
                    'Please provide an initial value or module to reduce the data.'
                )
            results.append(mod(results[-1], *cur_args, **cur_kwargs))
    if len(results) == 0:
        raise ValueError(
            'No results were produced from the reduction. '
            'Check that the input data is correct.'
        )
    return results[-1]

async def _execute_reduce_mod(mod, *args, **kwargs):
    """Execute the reduction module with the given args and kwargs

    Args:
        mod (AsyncProcess | Process): The module to execute
        *args: The arguments to pass to the module
        **kwargs: The keyword arguments to pass to the module
    Returns:
        typing.Any: The result of the module execution
    """
    print(mod)
    if isinstance(mod, AsyncProcess):
        return await mod.aforward(*args, **kwargs)
    elif isinstance(mod, Process):
        return mod.forward(*args, **kwargs)
    elif is_async_function(mod):
        return await mod(*args, **kwargs)
    else:
        return mod(*args, **kwargs)


async def async_reduce(
    mod: AsyncProcess | typing.Callable, 
    *args, 
    init_mod: AsyncProcess | typing.Callable=None, init_val=None, 
    **kwargs
) -> typing.Any:
    """Reduce the args passed in with a module

    Args:
        mod (Module): The module to use for reduction
        init_mod (Module, optional): The module to use for the first set of data. Defaults to None.
        init_val: The initial value to use
        
    Returns:
        The result of the reduction
    """
    results = []

    if init_val is not None:
        results.append(init_val)
    for _, cur_args, cur_kwargs in process_loop(None, *args, **kwargs):

        if len(results) == 0 and init_mod is not None and init_val is None:
            results.append(
                await _execute_reduce_mod(
                    init_mod, *cur_args, **cur_kwargs))
        elif len(results) == 0 and init_mod is not None:
            results.append(await _execute_reduce_mod(
                init_mod, init_val, *cur_args, **cur_kwargs)
            )
        else:
            if len(results) == 0:
                raise ValueError(
                    'No initial value or initial module provided. '
                    'Please provide an initial value or module to reduce the data.'
                )
            results.append(await _execute_reduce_mod(
                mod, results[-1], *cur_args, **cur_kwargs
            ))
    if len(results) == 0:
        raise ValueError(
            'No results were produced from the reduction. '
            'Check that the input data is correct.'
        )
    return results[-1]


class Partial(pydantic.BaseModel):
    """Class for storing a partial output from a streaming process
    """
    dx: typing.Any = None
    complete: bool = False
    prev: typing.Any = None
    full: typing.List = dataclasses.field(default_factory=list)

    def clone(self) -> 'Partial':
        """Clone the partial object

        Returns:
            Partial: The cloned partial object
        """
        return Partial(
            dx=self.dx, complete=self.complete, prev=self.prev, full=self.full.copy()
        )


class StreamSequence(StreamProcess, typing.Generic[PROCESS, STREAM]):
    """A process sequence that wraps a StreamProcess with a pre and post process.
    The pre and post processes are standard Processes that are applied before and after the streaming process.
    Each streamed item is postprocessed

    Example:
    process = StreamSequence(
        pre=pre_process,
        mod=streaming_model,
        post=post_process
    )
    """

    pre: PROCESS
    mod: STREAM
    post: PROCESS

    def stream(self, x: typing.Any) -> typing.Iterator:
        """Stream the input through the pre, mod, and post processes
        Args:
            x (typing.Any): The input to the process
        Yields:
            typing.Any: The output of the process
        """

        x = self.pre(x)
        for x_i in self.mod.stream(x):
            yield self.post(x_i)


class AsyncStreamSequence(AsyncStreamProcess, typing.Generic[PA, ASYNC_STREAM]):
    """Asynchronous stream sequence that applies a pre-processing, a module, and a post-processing step.

    Example:
    process = AsyncStreamSequence(
        pre=pre_process,
        mod=async_streaming_model,
        post=post_process
    )
    async for y in await process.astream(x):
        print(y)
    """

    pre: PA
    mod: ASYNC_STREAM
    post: PA

    async def astream(self, x: typing.Any) -> typing.AsyncIterator:

        if isinstance(self.pre, AsyncProcess):
            x = await self.pre.aforward(x)
        else:
            x = self.pre(x)
        async for x_i in self.mod.astream(x):
            if isinstance(self.post, AsyncProcess):
                yield await self.post.aforward(x_i)
            else:
                yield self.post(x_i)


class Func(Process):
    """Function process that applies a callable to the input data.
    """
    f: typing.Callable
    args: typing.List[typing.Any] = pydantic.Field(default_factory=list)
    kwargs: typing.Dict[str, typing.Any] = pydantic.Field(default_factory=dict)

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
    f: typing.Callable
    args: typing.List[typing.Any] = pydantic.Field(default_factory=list)
    kwargs: typing.Dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    async def forward(self, *args, **kwargs):

        return await self.f(
            *self.args, *args, **self.kwargs, **kwargs
        )
