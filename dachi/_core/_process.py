# 1st party
from abc import ABC
import typing
from typing import Any
import typing
import asyncio
from functools import wraps
from dataclasses import dataclass

# 3rd party
from dachi._core._core import Param
import numpy as np

# local
from ._core import Struct, Module

# 1st party
import typing
from abc import ABC, abstractmethod
from typing import get_type_hints
from typing import Self
import typing

from uuid import uuid4
from enum import Enum
import asyncio
from dataclasses import dataclass

import inspect
import json

# 3rd party
import pydantic

# local
from ._utils import (
    is_primitive, escape_curly_braces, 
    unescape_curly_braces,
    generic_class
)

from ._core import UNDEFINED



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


class StreamModule(Module):
    # Use for modules that rely on stream

    def forward(self, x: str) -> Any:
        
        out = None
        for out, c in self.stream_iter(x):
            pass
        return out


class ParallelModule(Module, ABC):
    """Module that allows 
    """

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    def __getitem__(self, idx: int) -> 'Module':

        return self._modules[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for module_i in self._modules:
            yield module_i


class Get(Module):

    def forward(self, struct: Struct, key) -> typing.Any:
        
        return struct[key]


class Set(Module):

    def forward(self, struct: Struct, key, value) -> typing.Any:
        
        struct[key] = value
        return value


get = Get()
set = Set()


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


class Multi(Module):
    
    def __init__(self, module: Module, split_args: typing.Set[str]) -> None:
        super().__init__()

        self.module = module
        self.split_args = split_args

    def forward(self, n_splits: int, **kwargs) -> Any:
        
        result = []
        for i in range(n_splits):
            kw = {}
            for k, v in kwargs.items():
                if k in self.split_args:
                    count = (len(v) // n_splits)
                    if i == n_splits - 1:
                        kw[k] = v[i * count:]
                    else:
                        kw[k] = v[i * count:(i + 1) * count]
                else:
                    kw[k] = v
            result.append(self.module(**kw))
        return result
    
    async def async_forward(self, n_splits: int, **kwargs) -> Any:
        
        results = []
        async with asyncio.TaskGroup() as tg:
            tasks = []
            for i in range(n_splits):
                kw = {}
                for k, v in kwargs.items():
                    if k in self.split_args:
                        count = (len(v) // n_splits)
                        if i == n_splits - 1:
                            kw[k] = v[i * count:]
                        else:
                            kw[k] = v[i * count:(i + 1) * count]
                    else:
                        kw[k] = v
                
                tasks.append(tg.create_task(self.module.async_forward(**kw)))
            for task in tasks:
                result = await task
                results.append(result)
        return result


class ModuleList(Module):

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    def children(self, recurse: bool = True) -> typing.Iterator[Module]:
        
        for module in self._modules:
            for child in module.children(recurse):
                yield child

    def parameters(self, recurse: bool = True) -> typing.Iterator[Param]:
        
        for module in self._modules:
            for p in module.parameters(recurse):
                yield p
        
    def forward(self) -> Any:
        return None



class Sequential(Module):
    
    def __init__(self, *module) -> None:
        super().__init__()

        self.modules = module

    def forward(self, *x) -> Any:
        
        first = False
        for module in self.modules:
            if first:
                x = module(*x)
            else:
                x = module(x)
        return x


class Batch(object):

    def __init__(self, data: typing.Iterable, n_samples: int, shuffle: bool, drop_last: bool=True):

        self.data = data
        self.shuffle = shuffle
        self.n_samples = n_samples
        self.drop_last = drop_last

    def __iter__(self) -> typing.Iterator[typing.Any]:

        if self.shuffle:
            order = np.random.permutation(len(self.data))
        else:
            order = np.linspace(0, len(self.data))
        
        n_iterations = len(self.data) // self.n_samples
        if len(self.data) % self.n_samples != 0 and not self.drop_last:
            n_iterations += 1

        start = 0
        upto = self.n_samples
        for _ in range(n_iterations):
            cur_data = self.data[order[start:upto]]
            yield cur_data
            start = upto
            upto += self.n_samples


@processf
def batchf(data: typing.Iterable, n_samples: int, shuffle: bool, drop_last: bool=True):
    
    return Batch(
        data, n_samples, shuffle, drop_last
    )


def stream(m: Module, *args, **kwargs) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:

    streamer = m.stream_forward(*args, **kwargs)

    for partial in streamer:
        yield partial.cur, partial.dx


class Module(ABC):
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

    def parameters(self, recurse: bool=True) -> typing.Iterator['Param']:
        """Loop over the parameters for the module

        Yields:
            Param: The parameters for the module
        """
        yielded = set()
        for k, v in self.__dict__.items():
            if isinstance(v, Param):
                if id(v) in yielded:
                    continue
                yielded.add(id(v))
                
                yield v
            if recurse and isinstance(v, Module):
                for v in v.parameters(True):
                    if id(v) in yielded:
                        continue
                    yielded.add(id(v))
                    yield v

    def children(self, recurse: bool=True) -> typing.Iterator['Module']:
        """Loop over all of the child modules

        Yields:
            Module: The child module
        """
        yielded = set()
        print(self.__dict__)
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                if id(v) in yielded:
                    continue
                yield v
                yielded.add(id(v))
                if recurse:
                    for v in v.children(True):
                        if id(v) in yielded:
                            continue
                        yielded.add(id(v))
                        yield v
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """Execute the forward method asynchronously

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)

    def stream_forward(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        """Stream the output

        Yields:
            Iterator[typing.Iterator[ typing.Tuple[typing.Any, typing.Any] ]]: The current value and the change in the value
        """
        # default behavior doesn't actually stream
        res = self.forward(*args, **kwargs) 
        yield res, res

    def streamer(self, *args, **kwargs) -> 'Streamer':
        """Retrieve a streamer

        Returns:
            Streamer: The Streamer to loop over
        """
        return Streamer(
            iter(self.stream_forward(*args, **kwargs))
        )

    def reduce(
        self,
        data: typing.Iterable, 
        *args, init=None, **kwargs
    ) -> typing.Any:
        """

        Args:
            data (typing.Iterable): 
            module (Module): 
            init (_type_, optional): . Defaults to None.

        Returns:
            typing.Any: 
        """
        cur = init
        for data_i in data:
            cur = self(cur, data_i, *args, **kwargs)
        return cur

    def map(self, data: typing.Iterable, *args, **kwargs) -> typing.Any:
        """

        Args:
            data (typing.Iterable): The data to map
            module (Module): The module to execute
            init : TODO: CHECK. Defaults to None.

        Returns:
            typing.Any: 
        """
        results = []
        for data_i in data:
            results.append(self(data_i, *args, **kwargs))
        return results

    async def async_map(self, data: typing.Iterable, *args, **kwargs):
        
        tasks: asyncio.Task = []
        async with asyncio.TaskGroup() as tg:
            for data_i in data:
                tasks.append(
                    tg.create_task(self, data_i, *args, **kwargs)
                )
        
            return tuple(task.result() for task in tasks)

    # async def async_stream_iter(self, *args, **kwargs) -> typing.AsyncIterator[
    #     typing.Tuple[typing.Any, typing.Any]
    # ]:
    #     # default behavior doesn't actually stream
    #     res = self.forward(*args, **kwargs) 
    #     yield res, None

    # async def async_stream_forward(self, *args, **kwargs) -> 'Streamer':
    #     """
    #     Returns:
    #         Streamer: The Streamer to loop over
    #     """
    #     return Streamer(
    #         self.stream_iter(*args, **kwargs)
    #     )

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

    def __init__(self, stream: typing.Iterator):
        """The Stream to loop over

        Args:
            stream: The stream to loop over in generating the stream
        """
        self._stream = stream
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
