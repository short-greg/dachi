# 1st party
import typing
from typing import Self, Any
import itertools
import time

from abc import ABC, abstractmethod
from abc import ABC
import asyncio
import threading
from enum import Enum

from dataclasses import dataclass

from functools import wraps
from uuid import uuid4
from enum import Enum

# 3rd party
import numpy as np
# 3rd party
import pydantic

# local
from ._core import Module
from dachi._core._core import Param
from ._core import UNDEFINED
from ._core import Renderable, render


@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


# class StructModule(Struct, Module):

#     def forward(self, key: str, value: typing.Any) -> typing.Any:
        
#         copy = self.model_copy()
#         copy[key] = value
#         return copy


class I(object):
    
    def __init__(self, data, n: int):
        self.data = data
        self.n = n

    def __iter__(self) -> typing.Iterator[typing.Any]:
        for i in range(self.n):
            yield self.data


class P(object):
    
    def __init__(self, data: typing.Iterable, n: int=None):

        self.data = data
        self._n = n or len(data)

    def __iter__(self) -> typing.Iterator:
        for d_i in self.data:
            yield d_i

    @classmethod
    def m(cls, *data: typing.Iterable, n: int=None) -> typing.Tuple['P']:

        return tuple(
            P(d, n) for d in data
        )
    
    @property
    def n(self) -> int:
        return self._n


def parallel_loop(modules: typing.Union['ModuleList', Module, None], *args, **kwargs) -> typing.Iterator[
    typing.Tuple[Module, typing.List, typing.Dict]
]:
    """

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

        self._modules = modules

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.List:
        pass


class MultiModule(ParallelModule):
    
    def forward(self, *args, **kwargs) -> typing.List:
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


class _ProcessMethod(Module):

    def __init__(self, f: typing.Callable, instance=None):
        self.f = f
        self.instance = instance
        self._stored = None
        self._async_f = None

    def forward(self, *args, **kwargs) -> typing.Any:
        if self.instance:
            return self.f(self.instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    async def async_forward(self, *args, **kwargs) -> typing.Any:
        
        if self._async_f:
            self._async_f(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):

        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = _ProcessMethod(self.f, instance)
        return self._stored
    
    @classmethod
    def async_(cls, f):

        cls._async_f = f


def processf(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return _ProcessMethod(f)
    else:
        return _ProcessMethod(wrapper)


class ModuleList(Module):

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    def children(self, recurse: bool = True) -> typing.Iterator[Module]:
        
        for module in self._modules:
            yield module
            for child in module.children(recurse):
                yield child

    def parameters(self, recurse: bool = True) -> typing.Iterator[Param]:
        
        for module in self._modules:
            for p in module.parameters(recurse):
                yield p
        
    def forward(self) -> Any:
        raise ValueError('Cannot pass forward with ModuleList')
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> typing.Iterator:

        for m in self._modules:
            yield m


class Sequential(ModuleList):
    
    def __init__(self, *module) -> None:
        super().__init__(module)

    def add(self, module):

        self._modules.append(module)

    def forward(self, *x) -> Any:
        
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
    
    def __len__(self) -> int:
        return len(self._modules)


class Batched(Renderable):

    def __init__(
        self, *data: typing.Iterable, 
        size: int=1, drop_last: bool=True, 
        order: typing.Optional[typing.List]=None
    ):

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
        return self._n

    def shuffle(self) -> Self:
        return Batched(
            *self._data, size=self._size, drop_last=self.drop_last, 
            order=np.random.permutation(self._n_elements)
        )
        # n_iterations = len(self._data) // self.size
        # if len(self._data) % self.size != 0 and not self.drop_last:
        #     n_iterations += 1

    def __iter__(self) -> typing.Iterator[typing.Any]:

        # if self.shuffle:
        #     order = np.random.permutation(self._n_elements)
        # else:
        #     order = np.linspace(0, len(self._n_elements))

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
        
        return '\n'.join(
            f'{i}: {render(v)}' for i, v in enumerate(self)
        )


def reduce(mod: Module, *args, init: Module=None, **kwargs):
    
    results = []

    for _, cur_args, cur_kwargs in parallel_loop(None, *args, **kwargs):

        if len(results) == 0 and init is not None:
            results.append(init(*cur_args, **cur_kwargs))
        elif len(results) == 0:
            results.append(mod(None, *cur_args, **cur_kwargs))
        else:
            results.append(mod(results[-1], *cur_args, **cur_kwargs))
    return results[-1]


async def _async_map(f: Module, *args, **kwargs) -> typing.Tuple[typing.Any]:

    tasks = []
    async with asyncio.TaskGroup() as tg:
        
        for cur_f, a, kv in parallel_loop(f, *args, **kwargs):

            tasks.append(tg.create_task(cur_f.async_forward(*a, **kv)))

    return tuple(
        t.result() for t in tasks
    )


def async_map(f: Module, *args, **kwargs) -> typing.Tuple[typing.Any]:
    
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
        return self._output is not UNDEFINED

    def __call__(self) -> typing.Union[Partial]:
        """Query the streamer and returned updated value if updated

        Returns:
            typing.Union[typing.Any, Partial]: Get the next value in the stream
        """
        if self._output is not UNDEFINED:
            return self._output
        try:
            self._prev = self._cur
            self._cur, self._dx = next(self._stream)
            return Partial(self._cur, self._prev, self._dx, False)    
        except StopIteration:
            self._output = Partial(self._cur, self._prev, self._dx, True) 
            return self._output
        
    def __iter__(self) -> typing.Iterator[Partial]:

        while True:

            cur = self()
            if cur.complete:
                break
            yield cur


class RunStatus(Enum):

    RUNNING = 'running'
    READY = 'ready'
    FINISHED = 'finished'


class Runner(object):

    def __init__(self, module, *args, **kwargs):
        
        self.module = module
        self.status = RunStatus.READY
        args = [module, *args]
        self._result = None
        self.t = threading.Thread(target=self._exec, args=args, kwargs=kwargs)
        self.t.run()

    @property
    def result(self) -> typing.Any:
        return self._result

    def _exec(self, module: Module, *args, **kwargs):

        self.status = RunStatus.RUNNING
        result = module(*args, **kwargs)
        self.status = RunStatus.FINISHED
        self._result = result


class StreamRunner(object):

    def __init__(self, module,  *args, **kwargs):
        
        self.module = module
        self.status = RunStatus.READY
        args = [module, *args]
        self._results = []
        self._result_dx = []
        self.t = threading.Thread(target=self._stream_exec, args=args, kwargs=kwargs)
        self.t.run()

    def _stream_exec(self, module: Module, *args, **kwargs):

        self.status = RunStatus.RUNNING
        for d, dx in module.stream_forward(*args, **kwargs):
            self._result_dx.append(dx)
            self._results.append(d)

        self.status = RunStatus.FINISHED

    @property
    def result(self) -> typing.Any:
        if len(self._results) > 0:
            return self._results[-1]
        return None
    
    def dx(self) -> typing.Iterator:
        for x in self._result_dx:
            yield x

    def exec_loop(self, sleep_time: float=1./60) -> typing.Iterator[typing.Tuple]:
        i = 0
        while True:

            if len(self._results) > i:
                yield self._results[i], self._result_dx[i]
                i += 1
            
            elif self.status == RunStatus.FINISHED:
                break
            
            time.sleep(sleep_time)

    def __iter__(self)  -> typing.Iterator[typing.Tuple]:

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
    """_summary_

    Args:
        module (_type_): _description_

    Returns:
        StreamRunner: _description_
    """

    return StreamRunner(
        module, *args, **kwargs
    )


# class Get(Module):

#     def forward(self, struct: Struct, key) -> typing.Any:
        
#         return struct[key]


# class Set(Module):

#     def forward(self, struct: Struct, key, value) -> typing.Any:
        
#         struct[key] = value
#         return value


# get = Get()
# set = Set()
