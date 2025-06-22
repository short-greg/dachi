# 1st party
from abc import ABC, abstractmethod
import asyncio
import typing
from typing import Any
import itertools
import asyncio

# 3rd party
import numpy as np
import pydantic

# local
from ..core import BaseModule, ModuleList
from ..utils import (
    is_async_function,
    is_generator_function
)
import dataclasses


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class Process(BaseModule):
    """Base class for Modules
    """
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


class AsyncProcess(BaseModule):
    """Base class for Modules
    """
    @abstractmethod
    async def aforward(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        pass


class StreamProcess(BaseModule):
    """Base class for Modules
    """   
    
    @abstractmethod
    def stream(self, *args, **kwargs) -> typing.Iterator[typing.Any]:
        """Stream the output

        Yields:
            Iterator[typing.Any]: The value streamed
        """
        pass


class AsyncStreamProcess(BaseModule):
    """Base class for Modules
    """   
    
    @abstractmethod
    async def astream(self, *args, **kwargs) -> typing.AsyncIterator:
        """
        Returns:
            Streamer: The Streamer to loop over
        """
        pass


def forward(
    f: typing.Union[Process, typing.Callable], *args, **kwargs
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
    f: typing.Union[Process, typing.Callable], *args, **kwargs
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
        return f(*args, **kwargs)
    if is_async_function(f) and not is_generator_function(f):
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
    elif not is_async_function(f) and is_generator_function(f):
        for v in f(*args, **kwargs):
            yield v
    elif is_async_function(f) and is_generator_function(f):
        raise NotImplementedError('Cannot execute an async streaming function from a streaming function')
    elif is_async_function(f) and not is_generator_function(f):
        raise NotImplementedError('Cannot stream with async function')
    elif not is_async_function(f) and not is_generator_function(f):
        yield f(*args, **kwargs)
    else:
        raise RuntimeError()


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
    elif is_async_function(f) and is_generator_function(f):
        async for v in await f(*args, **kwargs):
            yield v
    elif not is_async_function(f) and is_generator_function(f):
        for v in f(*args, **kwargs):
            yield v
    elif is_async_function(f) and not is_generator_function(f):
        yield await f(*args, **kwargs)
    elif not is_async_function(f) and not is_generator_function(f):
        yield f(*args, **kwargs)
    else: raise RuntimeError()


# TDOO: Doesn't serialize
class I(object):
    """Use to mark data that should not be batched when calling the map functions
    """
    
    def __init__(self, data, n: int):
        """Create an I object

        Args:
            data: 
            n (int): The number of times to loop
        """
        self.data = data
        self.n = n

    def __iter__(self) -> typing.Iterator[typing.Any]:
        """Iterate over the object (n times)

        Returns:
            typing.Iterator[typing.Any]: The 
        """
        for _ in range(self.n):
            yield self.data


class B(object):
    """Use to mark data for batching
    """
    def __init__(
        self, data: typing.Iterable, n: int=None
    ):
        """Create a P object that will loop over the data

        Args:
            data (typing.Iterable): The data to loop over
            n (int, optional): The number of items to loop over. Defaults to None.
        """
        self.data = data
        self._n = n or len(data)

    def __iter__(self) -> typing.Iterator:
        """

        Yields:
            typing.Any: Get each value in the  
        """
        for d_i in self.data:
            yield d_i

    @classmethod
    def m(
        cls, *data: typing.Iterable, n: int=None
    ) -> typing.Tuple['B']:
        """Wrap multiple data through Ps

        data: The data to wrap in P
        n: The number of batches to have

        Returns:
            typing.Tuple[P]: The resulting ps
        """
        return tuple(
            B(d, n) for d in data
        )
    
    @property
    def n(self) -> int:
        """Get the number of iterations

        Returns:
            int: The number of iterations in the loop
        """
        return self._n


class Sequential(ModuleList[Process]):
    """
    Sequential class wraps multiple modules into a sequential list of modules that will be executed one after the other.
    Methods:
        __init__(*module) -> None:
            Initialize the Sequential with a list of modules.
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
        for module in self._modules:
            if first and multi:
                x = module(*x)
            else:
                x = module(x)
            first = False
        return x

class AsyncFunc(AsyncProcess):
    """A function wrapper
    """
    f: typing.Callable
    args: typing.List[typing.Any]
    kwargs: typing.Dict[str, typing.Any]
    
    async def forward(self, *args, **kwargs):

        return await self.f(
            *self.args, *args, **self.kwargs, **kwargs
        )



class AsyncParallel(Process):
    """A type of Parallel module that makes use of 
    Python's Async
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
                self._modules, *args, **kwargs
            ):
                tasks.append(tg.create_task(
                    module.aforward(*cur_args, **cur_kwargs)
                ))

        return list(
            t.result() for t in tasks
        )


def process_loop(
    processes: typing.Union[typing.List[BaseModule], ModuleList, Process, None], *args, **kwargs
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
    if isinstance(processes, typing.List):
        processes = ModuleList(processes)

    sz = None
    if isinstance(processes, ModuleList):
        sz = len(processes)
    for arg in itertools.chain(args, kwargs.values()):
        if isinstance(arg, B):
            if sz is None:
                sz = arg.n
            elif arg.n != sz:
                raise ValueError()
    
    if sz is None:
        raise ValueError('None of the inputs can be parallelized')
    
    if not isinstance(processes, ModuleList):
        processes = I(processes, sz)

    kwarg_names = list(kwargs.keys())
    kwarg_vals = [I(v, sz) if not isinstance(v, B) else v for v in kwargs.values()]
    args = [I(v, sz) if not isinstance(v, B) else v for v in args]
    
    for vs in zip(processes, *args, *kwarg_vals):
        m = vs[0]
        args = vs[1:len(args) + 1]
        kwargs = vs[len(args) + 1:]
        yield m, args, dict(zip(kwarg_names, kwargs))


def create_task(
    tg: asyncio.TaskGroup,
    f: AsyncProcess | Process | typing.Callable,
    *args, 
    **kwargs
) -> typing.Any:

    if isinstance(f, AsyncProcess):
        return tg.create_task(f.aforward, *args, **kwargs)
    elif isinstance(f, Process):
        return tg.create_task(
            asyncio.to_thread(f.forward, *args, **kwargs))
    elif is_async_function(f):
        return tg.create_task(f, *args, **kwargs)
    return tg.create_task(
        tg.create_task(asyncio.to_thread(f, *args, *kwargs))
    )


def process_map(
    f: Process | typing.Callable, 
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

            if isinstance(cur_f, AsyncProcess):
                tasks.append(tg.create_task(cur_f.aforward, *a, **kv))
            else:
                tasks.append(tg.create_task(cur_f, *a, **kv))
            tasks.append(
                create_task(tg, cur_f, *a, **kv)
            )

    return tuple(
        t.result() for t in tasks
    )


def multiprocess(*f: Process | typing.Callable) -> typing.Tuple[typing.Any]:
    """Helper function to run asynchronous processes

    Returns:
        typing.Tuple[typing.Any]: 
    """
    return tuple(
        f_i() for f_i in f
    )


async def async_multiprocess(*f: AsyncProcess | typing.Callable) -> typing.Tuple[typing.Any]:
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

    for _, cur_args, cur_kwargs in process_loop(None, *args, **kwargs):

        if len(results) == 0 and init_mod is not None and init_val is None:
            results.append(init_mod(*cur_args, **cur_kwargs))
        elif len(results) == 0 and init_mod is not None:
            results.append(init_mod(init_val, *cur_args, **cur_kwargs))
        elif len(results) == 0:
            # print(init_val, cur_args, cur_kwargs)
            results.append(mod(init_val, *cur_args, **cur_kwargs))
        else:
            results.append(mod(results[-1], *cur_args, **cur_kwargs))
    return results[-1]


async def async_reduce(
    mod: AsyncProcess | typing.Callable, *args, init_mod: AsyncProcess | typing.Callable=None, init_val=None, **kwargs
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

    tasks = []
    if isinstance(init_mod, AsyncProcess):
        init_mod = init_mod.aforward
    if isinstance(mod, AsyncProcess):
        mod = mod.aforward

    for _, cur_args, cur_kwargs in process_loop(None, *args, **kwargs):

        if len(results) == 0 and init_mod is not None and init_val is None:

            results.append(await init_mod(*cur_args, **cur_kwargs))
        elif len(results) == 0 and init_mod is not None:
            results.append(await init_mod(init_val, *cur_args, **cur_kwargs))
        elif len(results) == 0:
            # print(init_val, cur_args, cur_kwargs)
            results.append(await mod(init_val, *cur_args, **cur_kwargs))
        else:
            results.append(await mod(results[-1], *cur_args, **cur_kwargs))
    return results[-1]


class Partial(pydantic.BaseModel):
    """Class for storing a partial output from a streaming process
    """
    dx: typing.Any = None
    complete: bool = False
    prev: typing.Any = None
    full: typing.List = dataclasses.field(default_factory=list)


class Func(Process):
    """A function wrapper
    """
    f: typing.Callable
    args: typing.List[typing.Any]
    kwargs: typing.Dict[str, typing.Any]

    def forward(self, *args, **kwargs):

        return self.f(
            *self.args, *args, **self.kwargs, **kwargs
        )
