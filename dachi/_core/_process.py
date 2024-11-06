# 1st party
from abc import ABC, abstractmethod
import typing
from typing import Self, Any
import itertools
import time
import asyncio
import threading
from enum import Enum
from dataclasses import dataclass
from functools import wraps

# 3rd party
import numpy as np

# local
from ._core import Module
from ._core import Param
from ..utils import UNDEFINED, Renderable
from ._core import render


@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class I(object):
    """Use to mark data that should not be batched
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


class P(object):
    """Use to mark data for batching
    """
    
    def __init__(self, data: typing.Iterable, n: int=None):
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
    def m(cls, *data: typing.Iterable, n: int=None) -> typing.Tuple['P']:
        """Wrap multiple data through Ps

        data: The data to wrap in P
        n: The number of batches to have

        Returns:
            typing.Tuple[P]: The resulting ps
        """
        return tuple(
            P(d, n) for d in data
        )
    
    @property
    def n(self) -> int:
        """Get the number of iterations

        Returns:
            int: The number of iterations in the loop
        """
        return self._n


def parallel_loop(modules: typing.Union['ModuleList', Module, None], *args, **kwargs) -> typing.Iterator[
    typing.Tuple[Module, typing.List, typing.Dict]
]:
    """Use to loop over the module list

    Args:
        modules (typing.Union[ModuleList, Module, None]): 

    Returns:
        typing.Iterator[ typing.Tuple[Module, typing.List, typing.Dict] ]: 

    Yields:
        Iterator[typing.Iterator[ typing.Tuple[Module, typing.List, typing.Dict] ]]: 
    """
    if isinstance(modules, typing.List):
        modules = ModuleList(modules)

    sz = None
    if isinstance(modules, ModuleList):
        sz = len(modules)
    for arg in itertools.chain(args, kwargs.values()):
        if isinstance(arg, P):
            if sz is None:
                sz = arg.n
            elif arg.n != sz:
                raise ValueError()
    
    if sz is None:
        raise ValueError('None of the inputs can be parallelized')
    
    if not isinstance(modules, ModuleList):
        modules = I(modules, sz)

    kwarg_names = list(kwargs.keys())
    kwarg_vals = [I(v, sz) if not isinstance(v, P) else v for v in kwargs.values()]
    args = [I(v, sz) if not isinstance(v, P) else v for v in args]
    
    for vs in zip(modules, *args, *kwarg_vals):
        m = vs[0]
        args = vs[1:len(args) + 1]
        kwargs = vs[len(args) + 1:]
        yield m, args, dict(zip(kwarg_names, kwargs))


class ParallelModule(Module, ABC):
    """Module that allows 
    """

    def __init__(self, modules: typing.Union[typing.List[Module], Module]):
        """

        Args:
            modules (typing.Union[typing.List[Module], Module]): 
        """
        self._modules = modules

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.List:
        """Execute the parallel module

        Returns:
            typing.List: The results of the parallel module
        """
        pass


class MultiModule(ParallelModule):
    """A module that executes each of the modules it wraps in a loop
    """
    
    def forward(self, *args, **kwargs) -> typing.List:
        """Loop over all of the modules and execute them

        Returns:
            typing.List: The output of all the modules
        """
        res = []
        for module, cur_args, cur_kwargs in parallel_loop(self._modules, *args, **kwargs):
           res.append(module(*cur_args, **cur_kwargs))
        return res


class AsyncModule(ParallelModule):
    """A type of Parallel module that makes use of 
    Python's Async
    """
    async def async_forward(self, *args, **kwargs) -> typing.List:
        """The asynchronous method to use for inputs

        Args:
            args (typing.List[Args]): The list of inputs to the modules 

        Returns:
            typing.Tuple: The output to the paralell module
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for module, cur_args, cur_kwargs in parallel_loop(self._modules, *args, **kwargs):
                tasks.append(tg.create_task(
                    module.async_forward(*cur_args, **cur_kwargs)
                ))

        return list(
            t.result() for t in tasks
        )

    def forward(self, *args, **kwargs) -> typing.Tuple:
        """Run the asynchronous module
        Returns:
            typing.Tuple: The output for the paralell module
        """
        return asyncio.run(self.async_forward(*args, **kwargs))


class ModuleList(Module):
    """Use to have a list of modules
    """

    def __init__(self, modules: typing.List[Module]):
        """Create a list of modules

        Args:
            modules (typing.List[Module]): the modules to set
        """
        self._modules = modules

    def children(self, recurse: bool = True) -> typing.Iterator[Module]:
        """Get the children of the ModuleList

        Args:
            recurse (bool, optional): Whether to recurse to child modules. Defaults to True.

        Yields:
            Module: The children of the module
        """
        for module in self._modules:
            yield module
            if recurse:
                for child in module.children(recurse):
                    yield child

    def parameters(self, recurse: bool = True) -> typing.Iterator[Param]:
        """Get the parameters in the ModuleList

        Args:
            recurse (bool, optional): _description_. Defaults to True

        Yields:
            Param: The params in the ModuleList
        """
        for module in self._modules:
            for p in module.parameters(recurse):
                yield p
        
    def forward(self) -> Any:
        """

        Raises:
            ValueError: If called because ModuleList cannot be __call__

        """
        raise ValueError('Cannot pass forward with ModuleList')
    
    def __len__(self) -> int:
        """Get the number of modules in the ModuleList

        Returns:
            int: The number of modules
        """
        return len(self._modules)
    
    def __iter__(self) -> typing.Iterator[Module]:
        """Loop over the modules in the ModuleList

        Yields:
            Module: All of the modules in the ModuleList
        """
        for m in self._modules:
            yield m


class Sequential(ModuleList):
    """Use to 
    """
    
    def __init__(self, *module) -> None:
        """Create a Sequence of modules
        """
        super().__init__(module)

    def add(self, module):
        """Add a module to Sequential

        Args:
            module: Add the module to the Sequential
        """
        self._modules.append(module)

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


class Batched(Renderable):
    """Batched is data 
    """

    def __init__(
        self, *data: typing.Iterable, 
        size: int=1, drop_last: bool=True, 
        order: typing.Optional[typing.List]=None
    ):
        """Create Batched data

        Args:
            size (int, optional): the size of each batch. Defaults to 1.
            drop_last (bool, optional): Whether to drop the last element. Defaults to True.
            order (typing.Optional[typing.List], optional): The order to return the values in. Defaults to None.
        """
        if len(data) == 0:
            raise ValueError('No data was passed in to batch')
        sz = None
        for v in data:
            if sz is None:
                sz = len(v)
            elif sz != len(v):
                raise ValueError(
                    'The lengths of all of the elements '
                    'to batch must be the same'
                )
        self._data = data
        self._size = size
        self._n_elements = sz
        self.drop_last = drop_last
        add_one = (self._n_elements % self._size) != 0
        self._n = (self._n_elements // self._size) + add_one
        self._order = (
            order if order is not None 
            else np.arange(self._n_elements)
        )

    def __len__(self) -> int:
        """Get the length of the batch

        Returns:
            int: The number of elements in the batch
        """
        return self._n

    def shuffle(self) -> Self:
        """Shuffle the batched data

        Returns:
            Self: Batched but shuffled
        """
        return Batched(
            *self._data, size=self._size, drop_last=self.drop_last, 
            order=np.random.permutation(self._n_elements)
        )

    def __iter__(self) -> typing.Iterator[typing.Any]:
        """Iterate over the batched data

        Yields:
            Iterator[typing.Iterator[typing.Any]]: _description_
        """
        start = 0
        upto = self._size
        for _ in range(self._n):
            cur = []
            for data in self._data:
                if isinstance(data, np.ndarray):
                    cur_data = data[self._order[start:upto]]
                else:
                    cur_data = [data[i] for i in self._order[start:upto]]
                cur.append(cur_data)
    
            start = upto
            upto += self._size
            if len(cur) == 1:
                yield cur[0]
            else:
                yield tuple(cur)
    
    def render(self) -> str:
        """Render the batch to a string

        Returns:
            str: The text
        """
        return '\n'.join(
            f'{i}: {render(v)}' for i, v in enumerate(self)
        )


def reduce(mod: Module, *args, init_mod: Module=None, init_val=None, **kwargs):
    """Reduce the args passed in with a module

    Args:
        mod (Module): The module to use for reduction
        init_mod (Module, optional): The module to use for the first set of data. Defaults to None.
        init_val: The initial value to use
        
    Returns:
        The result of the reduction
    """
    results = []

    for _, cur_args, cur_kwargs in parallel_loop(None, *args, **kwargs):

        if len(results) == 0 and init_mod is not None and init_val is None:
            results.append(init_mod(*cur_args, **cur_kwargs))
        elif len(results) == 0 and init_mod is not None:
            results.append(init_mod(init_val, *cur_args, **cur_kwargs))
        elif len(results) == 0:
            results.append(mod(init_val, *cur_args, **cur_kwargs))
        else:
            results.append(mod(results[-1], *cur_args, **cur_kwargs))
    return results[-1]


async def _async_map(f: Module, *args, **kwargs) -> typing.Tuple[typing.Any]:
    """Helper function to run async_map

    Args:
        f (Module): The function to asynchronously execute

    Returns:
        typing.Tuple[typing.Any]: The result of the map
    """
    tasks = []
    async with asyncio.TaskGroup() as tg:
        
        for cur_f, a, kv in parallel_loop(f, *args, **kwargs):

            tasks.append(tg.create_task(cur_f.async_forward(*a, **kv)))

    return tuple(
        t.result() for t in tasks
    )


def async_map(f: Module, *args, **kwargs) -> typing.Tuple[typing.Any]:
    """Map *args and **kwargs through the module

    Args:
        f (Module): The module to use for mapping

    Returns:
        typing.Tuple[typing.Any]: The output of the modules
    """
    return asyncio.run(_async_map(f, *args, **kwargs))


async def _async_multi(*f) -> typing.Tuple[typing.Any]:
    """Helper function to run asynchronous processes

    Returns:
        typing.Tuple[typing.Any]: 
    """

    tasks = []
    async with asyncio.TaskGroup() as tg:

        for f_i in f:
            tasks.append(tg.create_task(f_i))

    return tuple(
        task.result() for task in tasks
    )


def async_multi(*f) -> typing.Tuple[typing.Any]:
    """Run multiple asynchronous processes and return
    the results

    Args:
       f (coroutine): The coroutines to run

    Returns:
        typing.Tuple: The output of the 
    """
    return asyncio.run(_async_multi(*f))


@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class Streamer(object):
    """Streamer is an object used to stream over the response
    """

    def __init__(self, stream: typing.Iterable):
        """The Stream to loop over

        Args:
            stream: The stream to loop over in generating the stream
        """
        self._stream = iter(stream)
        self._cur = None
        self._output = UNDEFINED
        self._prev = None
        self._dx = None

    @property
    def complete(self) -> bool:
        """Check if the stream has completed

        Returns:
            bool: True if the stream is complete
        """
        return self._output is not UNDEFINED

    def __call__(self) -> Partial:
        """Query the streamer and returned updated value if updated

        Returns:
            typing.Union[typing.Any, Partial]: Get the next value in the stream
        """
        if self._output is not UNDEFINED:
            return self._output
        try:
            self._prev = self._cur
            self._cur, self._dx = next(self._stream)
            print('Cur = ', self._cur)
            return Partial(self._cur, self._prev, self._dx, False)    
        except StopIteration:
            self._output = Partial(self._cur, self._prev, self._dx, True) 
            return self._output
    
    @property
    def output(self) -> typing.Optional[Partial]:
        """Get the output of the streamer. Will be undefined if not complete

        Returns:
            typing.Any: The output of the Streamer
        """
        return self._output
        
    def __iter__(self) -> typing.Iterator[Partial]:
        """Iterate over the stream

        Yields:
            Partial: The partial results of the stream
        """
        while True:

            cur = self()
            if cur.complete:
                break
            yield cur


class RunStatus(Enum):
    """The status for a "Run"
    """

    RUNNING = 'running'
    READY = 'ready'
    FINISHED = 'finished'


class Runner(object):
    """Create an object to loop over the stream
    """

    def __init__(self, module, *args, **kwargs):
        """Create a runner which will loop over the stream

        Args:
            module: The module to stream
        """
        
        self.module = module
        self.status = RunStatus.READY
        args = [module, *args]
        self._result = None
        self.t = threading.Thread(target=self._exec, args=args, kwargs=kwargs)
        self.t.run()

    @property
    def result(self) -> typing.Any:
        """Return the result

        Returns:
            typing.Any: The result of the 
        """
        return self._result

    def _exec(self, module: Module, *args, **kwargs):
        """The method that gets executed by the thread

        Args:
            module (Module): The module to execute
        """
        self.status = RunStatus.RUNNING
        result = module(*args, **kwargs)
        self.status = RunStatus.FINISHED
        self._result = result


class StreamRunner(object):
    """Use to run a stream
    """

    def __init__(self, module,  *args, **kwargs):
        """Create a StreamRunner which will run the module in 
        a stream

        Args:
            module: The module to stream
        """
        self.module = module
        self.status = RunStatus.READY
        args = [module, *args]
        self._results = []
        self._result_dx = []
        self.t = threading.Thread(target=self._stream_exec, args=args, kwargs=kwargs)
        self.t.run()

    def _stream_exec(self, module: Module, *args, **kwargs):
        """Stream the module

        Args:
            module (Module): The module to run
        """

        self.status = RunStatus.RUNNING
        for d, dx in module.stream_forward(*args, **kwargs):
            self._result_dx.append(dx)
            self._results.append(d)

        self.status = RunStatus.FINISHED

    @property
    def result(self) -> typing.Any:
        """Get the result of the stream

        Returns:
            typing.Any: The result
        """
        if len(self._results) > 0:
            return self._results[-1]
        return None

    @property
    def result_dx(self) -> typing.Any:
        """Get the result of the stream

        Returns:
            typing.Any: The result
        """
        if len(self._result_dx) > 0:
            return [*self._result_dx]
        return None

    def dx(self) -> typing.Iterator:
        """Loop over the changes in the results

        Yields:
            typing.Any: Each change" in value
        """
        for x in self._result_dx:
            yield x

    def exec_loop(self, sleep_time: float=1./60) -> typing.Iterator[typing.Tuple]:
        """Execute the loop

        Args:
            sleep_time (float, optional): The interval to execute the loop at. Defaults to 1./60.

        Yields:
            typing.Tuple: Each result and dx
        """
        i = 0
        while True:
            if len(self._results) > i:
                yield self._results[i], self._result_dx[i]
                i += 1
            
            elif self.status == RunStatus.FINISHED:
                break
            
            time.sleep(sleep_time)

    def __iter__(self)  -> typing.Iterator[typing.Tuple]:
        """Loop over the results and the dxs

        Yields:
            typing.Tuple: The result and each change in the result
        """
        for r, rx in zip(self._results, self._result_dx):
            yield r, rx


def run_thread(module: Module, *args, **kwargs) -> Runner:
    """Convenience function to create a "Runner"

    Args:
        module (Module): The module to execute

    Returns:
        Runner: 
    """
    return Runner(
        module, *args, **kwargs
    )


def stream_thread(module, *args, **kwargs) -> StreamRunner:
    """Convenience function to stream a thread. Will create a StreamRunner

    Args:
        module: The module to run

    Returns:
        StreamRunner: The stream runner
    """
    return StreamRunner(
        module, *args, **kwargs
    )
