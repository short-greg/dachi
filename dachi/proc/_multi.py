
# 1st party
from abc import ABC, abstractmethod
from functools import partial
import asyncio
import itertools
import dataclasses
import numpy as np
import typing as t

# 3rd party
import pydantic

# local
from ..core import ModuleList
from ..utils import (
    is_async_function,

)
from ._process import (
    Process,
    PA,
    AsyncProcess,
    StreamProcess,
    AsyncStreamProcess,
    PROCESS,
    STREAM,
    ASYNC_STREAM,
)


class Chunk(pydantic.BaseModel):
    """Use to mark data for batching
    As opposed to Recur, Chunk will batch the data into n chunks

    Example:
    chunk = Chunk(data=[1,2,3,4,5], n=2)
    for c in chunk:
        print(c)
        # prints [1,2] then [3,4,5]
    """
    data: t.List
    n: int | None = None
    shuffle: bool = False
    use_last: bool = True

    def __iter__(self) -> t.Iterator[t.List]:
        """

        Yields:
            t.Any: Get each value in the  
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
        cls, *data: t.Iterable, n: int=None
    ) -> t.Tuple['Chunk']:
        """Wrap multiple data through Ps

        data: The data to wrap in P
        n: The number of batches to have

        Returns:
            t.Tuple[P]: The resulting ps
        """
        return tuple(
            Chunk(data=d, n=n) for d in data
        )
    


# TDOO: Doesn't serialize
class Recur(pydantic.BaseModel):
    """Use to mark data that should not be batched when calling the map functions

    Example:
    recur = Recur(data="hello", n=5)
    for r in recur:
        print(r)
        # prints "hello" 5 times
    """
    data: t.Any
    n: int

    @pydantic.field_validator('n')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('n must be greater than 0')
        return v

    def __iter__(self) -> t.Iterator[t.Any]:
        """Iterate over the object (n times)

        Returns:
            t.Iterator[t.Any]: The 
        """
        for _ in range(self.n):
            yield self.data


def chunk(
    *data: t.List, n: int=None
) -> t.Tuple[Chunk] | Chunk:
    """Wrap multiple data through Ps
    data: The data to wrap in P
    n: The number of batches to have
    Returns:
        t.Tuple[Chunk]: The resulting ps
    """
    if len(data) == 1:
        return Chunk(data=data[0], n=n)
    return tuple(
        Chunk(data=d, n=n) for d in data
    )


def recur(
    *data: t.Any, n: int
) -> t.Tuple[Recur] | Recur:
    """Wrap multiple data through Ps
    data: The data to wrap in P
    n: The number of iterations to have
    Returns:
        t.Tuple[Recur]: The resulting ps
    """
    if len(data) == 1:
        return Recur(data=data[0], n=n)
    return tuple(
        Recur(data=d, n=n) for d in data
    )


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
    def forward(self, *x) -> t.Any:
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
    ModuleList[PA], AsyncProcess
):
    """Calls multiple modules in parallel and then 
    combines the results into a list

    Example:
    process = AsyncParallel(
        items=[process1, process2, process3]
    )
    y1, y2, y3 = process.aforward(x)
    """
    async def aforward(self, *args, **kwargs) -> t.List:
        """The asynchronous method to use for inputs

        Args:
            args (t.List[Args]): The list of inputs to the modules

        Returns:
            t.Tuple: The output to the paralell module
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for module, cur_args, cur_kwargs in process_loop(
                self.vals, *args, **kwargs
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
    processes: t.List | None, *args, **kwargs
) -> t.Iterator[
    t.Tuple[Process, t.List, t.Dict]
]:
    """Use to loop over the module list

    Args:
        modules (t.Union[ModuleList, Module, None]): 

    Returns:
        t.Iterator[ t.Tuple[Module, t.List, t.Dict] ]: 

    Yields:
        Iterator[t.Iterator[ t.Tuple[Module, t.List, t.Dict] ]]: 
    """
    # if isinstance(processes, t.List):
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
    f: AsyncProcess | Process | t.Callable,
    *args, 
    **kwargs
) -> t.Any:
    """Create a task for the process to run in the task group

    This function creates a task in the provided asyncio TaskGroup for the given function `f`.
    It handles both synchronous and asynchronous functions, as well as Process and AsyncProcess types.
    
    Args:
        tg (asyncio.TaskGroup): The task group to create the task in
        f (AsyncProcess | Process | t.Callable): The function to run
        *args: The arguments to pass to the function
        **kwargs: The keyword arguments to pass to the function
        
    Returns:

        t.Any: The task created
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
    f: Process | AsyncProcess | t.Callable, 
    *args, **kwargs
) -> t.Tuple[t.Any]:
    """Helper function to run async_map

    Args:
        f (Module): The function to asynchronously execute

    Returns:
        t.Tuple[t.Any]: The result of the map
    """
    return tuple(
        cur_f(*a, **kv) for cur_f, a, kv in process_loop(f, *args, **kwargs)
    )


async def async_process_map(
    f: AsyncProcess | t.Callable, 
    *args, **kwargs
) -> t.Tuple[t.Any]:
    """Helper function to run async_map

    Args:
        f (Module): The function to asynchronously execute

    Returns:
        t.Tuple[t.Any]: The result of the map
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
    *f: Process | t.Callable
) -> t.Tuple[t.Any]:
    """Helper function to run asynchronous processes

    Returns:
        t.Tuple[t.Any]: 
    """
    return tuple(
        f_i() for f_i in f
    )


async def async_multiprocess(
    *f: AsyncProcess | t.Callable
) -> t.Tuple[t.Any]:
    """Helper function to run asynchronous processes

    Returns:
        t.Tuple[t.Any]: 
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
) -> t.Any:
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
        t.Any: The result of the module execution
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
    mod: AsyncProcess | t.Callable, 
    *args, 
    init_mod: AsyncProcess | t.Callable=None, init_val=None, 
    **kwargs
) -> t.Any:
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
    dx: t.Any = None
    complete: bool = False
    prev: t.Any = None
    full: t.List = dataclasses.field(default_factory=list)

    def clone(self) -> 'Partial':
        """Clone the partial object

        Returns:
            Partial: The cloned partial object
        """
        return Partial(
            dx=self.dx, complete=self.complete, prev=self.prev, full=self.full.copy()
        )


class StreamSequence(StreamProcess, t.Generic[PROCESS, STREAM]):
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

    def stream(self, x: t.Any) -> t.Iterator:
        """Stream the input through the pre, mod, and post processes
        Args:
            x (t.Any): The input to the process
        Yields:
            t.Any: The output of the process
        """

        x = self.pre(x)
        for x_i in self.mod.stream(x):
            yield self.post(x_i)


class AsyncStreamSequence(AsyncStreamProcess, t.Generic[PA, ASYNC_STREAM]):
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

    async def astream(self, x: t.Any) -> t.AsyncIterator:

        if isinstance(self.pre, AsyncProcess):
            x = await self.pre.aforward(x)
        else:
            x = self.pre(x)
        async for x_i in self.mod.astream(x):
            if isinstance(self.post, AsyncProcess):
                yield await self.post.aforward(x_i)
            else:
                yield self.post(x_i)
