# 1st party
import typing
from typing import Self, Any
import itertools

from abc import ABC, abstractmethod
from abc import ABC
import asyncio

from dataclasses import dataclass

from functools import wraps
from uuid import uuid4
from enum import Enum

# 3rd party
import numpy as np
# 3rd party
import pydantic

# local
from ._core import Struct, Module, Renderable
from dachi._core._core import Param
from ._core import UNDEFINED
from ._utils import Args
from ._process import Module

# TODO: Decide how to handle this.. I want to try
# and get a more robust approach to handling these
# processes


# 1) pass in a list to multiple modules (N/N)
# 2) pass in a list to a single module (N/1)
# 3) pass in a single item to multiple modules (1/N)
# specify which it is in advance


# Some data I want to pass is static
# Some needs to be "split"

# parallel(*args, **kwargs)
# Args(*args, **kwargs)

# Parallel() -

# 
# paralell()


# Batch
#   1) returns a single item (size=None)
#   2) returns multiple items (list, size=a number)

# Parallelize(Module)
#   def forward(self,)

@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class StructModule(Struct, Module):

    def forward(self, key: str, value: typing.Any) -> typing.Any:
        
        copy = self.model_copy()
        copy[key] = value
        return copy


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


def parallel_loop(modules: typing.Union['ModuleList', Module], *args, **kwargs) -> typing.Iterator[
    typing.Tuple[Module, typing.List, typing.Dict]
]:
    if isinstance(modules, list):
        modules = ModuleList(modules)

    sz = None
    if isinstance(modules, ModuleList):
        sz = len(modules)
    for arg in itertools.chain(args, kwargs.values()):
        print('ARG: ', arg)
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


# TODO: finish this

# class Parallel(object):

#     def __init__(self, modules: typing.Union['ModuleList', Module], *args, **kwargs):

#         if isinstance(modules, typing.List):
#             modules = ModuleList(modules)
#         self.modules = P(modules) if isinstance(modules, ModuleList) else I(modules)
#         self.args = [arg if isinstance(arg, P) else I(arg) for arg in args]
#         self.kwargs = {k: arg if isinstance(arg, P) else I(arg) for k, arg in kwargs.items()}



# class Get(Module):

#     def forward(self, struct: Struct, key) -> typing.Any:
        
#         return struct[key]


# class Set(Module):

#     def forward(self, struct: Struct, key, value) -> typing.Any:
        
#         struct[key] = value
#         return value


# get = Get()
# set = Set()


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

# Batched()

# batched.reduce(lambda x, y: ...)
# batched.map(lambda x, y: ...)
# Batched(...).reduce()
# Batched(x=.., y=...).reduce()


class Batched(object):

    def __init__(self, *data: typing.Iterable, size: int=1, drop_last: bool=True, order: typing.Optional[typing.List]=None):

        if len(data) == 0:
            raise ValueError('No data was passed in to batch')
        sz = None
        for v in data:
            if sz is None:
                sz = len(v)
            elif sz != len(v):
                raise ValueError('The lengths of all of the elements to batch must be the same')
        self._data = data
        self._size = size
        self._n_elements = sz
        self.drop_last = drop_last
        add_one = (self._n_elements % self._size) != 0
        self._n = (self._n_elements // self._size) + add_one
        self._order = order if order is not None else np.arange(self._n_elements)

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

    # def map(self, f: typing.Callable, *args, **kwargs) -> typing.List:
        
    #     results = []
    #     for data in self:
    #         results.append(f(data, *args, **kwargs))
    #     return results
    
    # async def _async_map_helper(self, f, *args,  **kwargs) -> typing.List:
        
    #     async with asyncio.TaskGroup() as tg:
    #         tasks = []
    #         for data in self:
    #             tasks.append(tg.create_task(f, data, *args, **kwargs))
        
    #         return [task.result() for task in tasks]

    # def async_map(self, f, *args,  **kwargs) -> typing.List:
        
    #     with asyncio.Runner() as run:
    #         result = run.run(self._async_map_helper(f, *args, **kwargs))
    #     return result


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


# class Multi(ParallelModule):
#     """A type of Parallel module that simply
#     makes multiple calls
#     """
#     def forward(self, *args: Args) -> typing.Tuple:
#         """

#         Returns:
#             typing.Tuple: 
#         """
#         return tuple(
#             module_i.forward(*arg_i.args, **arg_i.kwargs)
#             for module_i, arg_i in zip(self._modules, args)
#         )


# class Multi(Module):
    
#     def __init__(self, module: Module, split_args: typing.Set[str]) -> None:
#         super().__init__()

#         self.module = module
#         self.split_args = split_args

#     def forward(self, n_splits: int, **kwargs) -> Any:
        
#         result = []
#         for i in range(n_splits):
#             kw = {}
#             for k, v in kwargs.items():
#                 if k in self.split_args:
#                     count = (len(v) // n_splits)
#                     if i == n_splits - 1:
#                         kw[k] = v[i * count:]
#                     else:
#                         kw[k] = v[i * count:(i + 1) * count]
#                 else:
#                     kw[k] = v
#             result.append(self.module(**kw))
#         return result
    
#     async def async_forward(self, n_splits: int, **kwargs) -> Any:
        
#         results = []
#         async with asyncio.TaskGroup() as tg:
#             tasks = []
#             for i in range(n_splits):
#                 kw = {}
#                 for k, v in kwargs.items():
#                     if k in self.split_args:
#                         count = (len(v) // n_splits)
#                         if i == n_splits - 1:
#                             kw[k] = v[i * count:]
#                         else:
#                             kw[k] = v[i * count:(i + 1) * count]
#                     else:
#                         kw[k] = v
                
#                 tasks.append(tg.create_task(self.module.async_forward(**kw)))
#             for task in tasks:
#                 result = await task
#                 results.append(result)
#         return result


# def parallel(m: Module, *args, **kwargs) -> typing.List:
#     pass


# paralelize()

# number per minibatch
# the elements

# def reduce(
#     module: Module,
#     data: typing.Iterable, 
#     *args, init=None, **kwargs
# ) -> typing.Any:
#     """

#     Args:
#         data (typing.Iterable): 
#         module (Module): 
#         init (_type_, optional): . Defaults to None.

#     Returns:
#         typing.Any: 
#     """
#     cur = init
#     for data_i in data:
#         cur = module(cur, data_i, *args, **kwargs)
#     return cur


# def map(module: Module, data: typing.Iterable, *args, **kwargs) -> typing.Any:
#     """

#     Args:
#         data (typing.Iterable): The data to map
#         module (Module): The module to execute
#         init : TODO: CHECK. Defaults to None.

#     Returns:
#         typing.Any: 
#     """
#     results = []
#     for data_i in data:
#         results.append(module(data_i, *args, **kwargs))
#     return results


# async def async_map(module: Module, data: typing.Iterable, *args, **kwargs):
    
#     tasks: asyncio.Task = []
#     async with asyncio.TaskGroup() as tg:
#         for data_i in data:
#             tasks.append(
#                 tg.create_task(module, data_i, *args, **kwargs)
#             )
    
#         return tuple(task.result() for task in tasks)



# @processf
# def reduce(
#     data: typing.Iterable, 
#     module: Module, 
#     *args, init=None, **kwargs
# ) -> typing.Any:
#     """

#     Args:
#         data (typing.Iterable): 
#         module (Module): 
#         init (_type_, optional): . Defaults to None.

#     Returns:
#         typing.Any: 
#     """
#     cur = init
#     for data_i in data:
#         cur = module(cur, data_i)
#     return cur


# @processf
# def map(data: typing.Iterable, module: Module, *args, init=None, **kwargs) -> typing.Any:
#     """

#     Args:
#         data (typing.Iterable): The data to map
#         module (Module): The module to execute
#         init : TODO: CHECK. Defaults to None.

#     Returns:
#         typing.Any: 
#     """
#     results = []
#     for data_i in data:
#         results.append(module(data_i, *args, **kwargs))
#     return results


# # TODO: figure out how to do this
# @map.async_
# async def async_map(data: typing.Iterable, module: Module, *args, **kwargs):
    
#     tasks: asyncio.Task = []
#     async with asyncio.TaskGroup() as tg:
#         for data_i in data:
#             tasks.append(
#                 tg.create_task(module, data_i, *args, **kwargs)
#             )
    
#         return tuple(task.result() for task in tasks)



# class Module(ABC):
#     """Base class for Modules
#     """

#     @abstractmethod
#     def forward(self, *args, **kwargs) -> typing.Any:
#         pass

#     def __call__(self, *args, **kwargs) -> typing.Any:

#         return self.forward(*args, **kwargs)

#     def parameters(self, recurse: bool=True) -> typing.Iterator[Param]:
        
#         yielded = set()
#         for k, v in self.__dict__.items():
#             if isinstance(v, Param):
#                 if id(v) in yielded:
#                     continue
#                 yielded.add(id(v))
                
#                 yield v
#             if recurse and isinstance(v, Module):
#                 for v in v.parameters(True):
#                     if id(v) in yielded:
#                         continue
#                     yielded.add(id(v))
#                     yield v

#     def children(self, recurse: bool=True) -> typing.Iterator['Module']:
        
#         yielded = set()
#         for k, v in self.__dict__.items():
#             if isinstance(v, Module):
#                 if id(v) in yielded:
#                     continue
#                 yielded.add(id(v))
#                 if recurse:
#                     for v in v.children(True):
#                         if id(v) in yielded:
#                             continue
#                         yielded.add(id(v))
#                         yield v
    
#     async def async_forward(self, *args, **kwargs) -> typing.Any:
#         """

#         Returns:
#             typing.Any: 
#         """
#         return self.forward(*args, **kwargs)

#     def stream_iter(self, *args, **kwargs) -> typing.Iterator[
#         typing.Tuple[typing.Any, typing.Any]
#     ]:
#         # default behavior doesn't actually stream
#         res = self.forward(*args, **kwargs) 
#         yield res, None

#     def stream_forward(self, *args, **kwargs) -> Streamer:
#         """
#         Returns:
#             Streamer: The Streamer to loop over
#         """
#         return Streamer(
#             self.stream_iter(*args, **kwargs)
#         )


# class Streamer(object):

#     def __init__(self, stream: typing.Iterator):
#         """The Stream to loop over

#         Args:
#             stream: The stream to loop over in generating the stream
#         """
#         self._stream = stream
#         self._cur = None
#         self._output = UNDEFINED
#         self._prev = None
#         self._dx = None

#     @property
#     def complete(self) -> bool:
#         return self._output is not UNDEFINED

#     def __call__(self) -> typing.Union[typing.Any, Partial]:
#         """Query the streamer and returned updated value if updated

#         Returns:
#             typing.Union[typing.Any, Partial]: Get the next value in the stream
#         """
#         if self._output is not UNDEFINED:
#             return self._output
#         try:
#             self._prev = self._cur
#             self._cur, self._dx = next(self._stream)
#             return Partial(self._cur, self._prev, self._dx, False)    
#         except StopIteration:
#             self._output = Partial(self._cur, self._prev, self._dx, True) 
#             return self._output
        
#     def __iter__(self) -> typing.Iterator[Partial]:

#         while True:

#             cur = self()
#             if cur.complete:
#                 break
#             yield cur
# class StreamModule(Module):
#     # Use for modules that rely on stream

#     def forward(self, x: str) -> Any:
        
#         out = None
#         for out, c in self.stream_iter(x):
#             pass
#         return out
