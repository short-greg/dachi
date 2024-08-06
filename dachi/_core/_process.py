# 1st party
from abc import abstractmethod, ABC
import typing
import time
from typing import Any
import asyncio
from functools import wraps
from dataclasses import dataclass

# local
from ._core import Param, Struct
from ._core import UNDEFINED


@dataclass
class Partial(object):
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class Streamer(object):

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

    def __call__(self) -> typing.Union[typing.Any, Partial]:
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


class Module(ABC):
    """Base class for Modules
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:

        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool=True) -> typing.Iterator[Param]:
        
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
        
        yielded = set()
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                if id(v) in yielded:
                    continue
                yielded.add(id(v))
                if recurse:
                    for v in v.children(True):
                        if id(v) in yielded:
                            continue
                        yielded.add(id(v))
                        yield v
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)

    def stream_iter(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        # default behavior doesn't actually stream
        yield self.forward(*args, **kwargs) 

    def stream_forward(self, *args, **kwargs) -> Streamer:
        """
        Returns:
            Streamer: The Streamer to loop over
        """
        return Streamer(
            self.stream_iter(*args, **kwargs)
        )

class StreamModule(Module):
    # Use for modules that rely on stream

    def forward(self, x: str) -> Any:
        
        out = None
        for out, c in self.stream_iter(x):
            pass
        return out


# class StreamableModule(Module, ABC):
#     """Module that defines a "stream_iter" method which
#     allows for streaming outputs
#     """

    # @abstractmethod
    # def stream_iter(self, *args, **kwargs) -> typing.Iterator[
    #     typing.Tuple[typing.Any, typing.Any]
    # ]:
    #     pass 

    # def forward(self, *args, **kwargs) -> Streamer:
    #     """
    #     Returns:
    #         Streamer: The Streamer to loop over
    #     """
    #     return Streamer(
    #         self.stream_iter(*args, **kwargs)
    #     )


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


class StructModule(Struct, Module):

    def forward(self, key: str, value: typing.Any) -> typing.Any:
        
        copy = self.model_copy()
        copy[key] = value
        return copy


class Get(Module):

    def forward(self, struct: Struct, key) -> typing.Any:
        
        return struct[key]


class Set(Module):

    def forward(self, struct: Struct, key, value) -> typing.Any:
        
        struct[key] = value
        return value


get = Get()
set = Set()


class _DecMethod(Module):

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
        self._stored = _DecMethod(self.f, instance)
        return self._stored
    
    @classmethod
    def async_(cls, f):

        cls._async_f = f


def processf(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return _DecMethod(f)
    else:
        return _DecMethod(wrapper)


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


# class WaitSrc(Src):
#     """Indicates to wait until completed
#     """

#     def __init__(self, incoming: T):
#         """

#         Args:
#             incoming (T): 
#         """
#         super().__init__()
#         self._incoming = incoming

#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: The incoming Transmission
#         """
#         yield self._incoming

#     def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
#         """
#         Args:
#             by (typing.Dict[T, typing.Any]): The input to the network

#         Returns:
#             typing.Any: The output of the Src
#         """
#         if isinstance(self._incoming.val, Partial) and not self._incoming.val.complete:
#             return WAITING

#         if isinstance(self._incoming.val, Streamer) and not self._incoming.val.complete:
#             return WAITING
#         return self._incoming.val
    

# def wait(t: T) -> T:
#     """Specify to wait for a streamed Transmission to complete before executing

#     Args:
#         t (T): The transmission

#     Returns:
#         T: The T to wait for the output
#     """

#     if isinstance(t.val, Partial) or isinstance(t.val, Streamer):
#         val = WAITING
#     else:
#         val = t.val
    
#     return T(val, WaitSrc(t))


# def stream(module: 'StreamableModule', *args, interval: float=None, **kwargs) -> typing.Iterator[T]:
#     """Use to loop over a streamable module until complete

#     Args:
#         module (StreamableModule): The module to stream over
#         interval (float, optional): The interval to stream over. Defaults to None.

#     Raises:
#         RuntimeError: If the module is not "Streamable"

#     Yields:
#         Iterator[typing.Iterator[T]]: _description_
#     """
#     if not isinstance(module, StreamableModule):
#         raise RuntimeError('Stream only works for streamable modules')
#     t = module.link(*args, **kwargs)
#     yield t

#     if isinstance(t.val, Streamer):
#         while not t.val.complete:
#             if interval is not None:
#                 time.sleep(interval)
#             yield t
