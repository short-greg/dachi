# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self

# 3rd party
import networkx as nx
import functools
import time
from enum import Enum

from dataclasses import dataclass
import uuid


class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'

UNDEFINED = _Types.UNDEFINED
WAITING = _Types.WAITING


def is_undefined(val):

    return val is UNDEFINED or val is WAITING


class Src(ABC):

    @abstractmethod
    def incoming(self) -> typing.Iterator['T']:
        pass

    @abstractmethod
    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        pass

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        return self.forward(by)


class StreamSrc(object):

    def __init__(self, module: 'StreamableModule', args: 'Args') -> None:
        super().__init__()

        self._module = module
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        
        for incoming in self._args.incoming:
            yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:

        if self in by:
            value: Streamer = by[self]
            return value
    
        args = self._args.iterate(by)        
        streamer = by[self] = self._module.forward(*args.args, **args.kwargs)
        return streamer
    
    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        return self.forward(by)


@dataclass
class Partial(object):

    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class T(object):

    def __init__(
        self, val=UNDEFINED, src: Src=None,
        name: str=None, annotation: str=None
    ):
        """

        Args:
            val (optional): . Defaults to UNDEFINED.
            src (Src, optional): . Defaults to None.
            name (str, optional): . Defaults to None.
            annotation (str, optional): . Defaults to None.
        """
        self._val = val
        self._src = src
        self._name = name
        self._annnotation = annotation

    @property
    def src(self) -> 'Src':

        return self._src

    # Move up a level
    def label(self, name: str=None, annotation: str=None) -> Self:

        if name is not None:
            self._name = name
        if annotation is not None:
            self._annnotation = annotation

    @property
    def val(self) -> typing.Any:
        return self._val

    def is_undefined(self) -> bool:

        return self._val is UNDEFINED

    def __getitem__(self, idx: int) -> 'T':

        if is_undefined(self._val):
            return T(
                self._val, IdxSrc(self, idx)
            )

        return T(
            self._val[idx], IdxSrc(self, idx)
        )
    
    def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        if not is_undefined(self._val):
            return self._val

        if self in by and not isinstance(by[self], Streamer) and not isinstance(by[self], Partial):
            return by[self]
    
        if self._src is not None:
            for incoming in self._src.incoming():
                incoming.probe(by)
            val = by[self] = self.src(by)
            if isinstance(val, Streamer):
                val = val()
            return val
        
        raise RuntimeError('Val has not been defined and no source for t')

    def detach(self):
        return T(
            self._val, None
        )


class Var(Src):
    
    def __init__(self, default=None, default_factory=None):

        self.default = default
        self.default_factory = default_factory

    def incoming(self) -> typing.Iterator[T]:

        # hack to ensure it is a generator
        if False:
            yield False
        
    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by.get[self]
        if self.default is not None:
            return self.default
        if self.default_factory is not None:
            return self.default_factory()
        
        raise RuntimeError('')


class IdxSrc(Src):

    def __init__(self, t: T, idx):

        self.t = t
        self.idx = idx

    def incoming(self) -> typing.Iterator['T']:
        yield self.t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by[self]
        val = self.t.probe(by)
        if is_undefined(val):
            return val
        return val[self.idx]


class Args(object):

    def __init__(self, *args, **kwargs):
        
        undefined = False

        for arg in args:
            
            if isinstance(arg, T):
                if is_undefined(arg.val):
                    undefined = True
                    break
                
        for k, arg in kwargs.items():
            
            if isinstance(arg, T):
                if is_undefined(arg.val):
                    undefined = True
                    break
        self._args = args
        self._undefined = undefined
        self._kwargs = kwargs
    
    def is_undefined(self) -> bool:
        return self._undefined
    
    def eval(self) -> Self:

        if self._undefined:
            return None
        args = []
        kwargs = {}
        for a in self._args:
            if isinstance(a, T):
                a = a.val
            if isinstance(a, Streamer):
                a = a().cur
            if isinstance(a, Partial):
                a = a.cur
            args.append(a)
        
        for k, a in self._kwargs.items():
            if isinstance(a, T):
                a = a.val
            if isinstance(a, Streamer):
                a = a()
            if isinstance(a, Partial):
                a = a.cur
            kwargs[k] = a
        return Args(*args, **kwargs)
    
    @property
    def args(self) -> typing.List:

        return self._args
    
    @property
    def kwargs(self) -> typing.Dict:
        return self._kwargs
    
    def incoming(self) -> typing.Iterator['T']:

        for arg in self._args:
            if isinstance(arg, T):
                yield arg

        for k, arg in self._kwargs:
            if isinstance(arg, T):
                yield arg

    def has_partial(self) -> bool:

        for a in self._args:
            if isinstance(a, T):
                a = a.val
            if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
                
                return True
        for k, a in self._kwargs.items():
            if isinstance(a, T):
                a = a.val
            if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
                return True
        return False
    
    def forward(self, by: typing.Dict['T', typing.Any]=None) -> Self:

        by = by or {}
        args = []
        kwargs = {}
        for arg in self._args:
            if isinstance(arg, T) and arg in by:
                val = by[arg]
                if isinstance(val, Partial):
                    # partial = True
                    args.append(val.cur)
                else:
                    args.append(val)
            else:
                args.append(arg)
            
        for k, arg in self._kwargs.items():
            if isinstance(arg, T) and arg in by:
                kwargs[k] = by[arg]
            else:
                kwargs[k] = arg
        
        return Args(*args, **kwargs)
    
    def iterate(self, by: typing.Dict['T', typing.Any], interval=0.01) -> Self:

        args = self
        while args.has_partial():
            args = args(by)
            time.sleep(interval)
        return args
        
    def __call__(self, by: typing.Dict['T', typing.Any]) -> Self:
        return self.forward(by)


# use partial goes in the module

class FSrc(Src):

    def __init__(self, mod: 'Module', args: Args):

        super().__init__()
        self.mod = mod
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        
        for t in self._args.incoming():
            yield t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        # what if "by" is partial
        if self in by:
            return by[self]
        
        if not self._args.has_partial():
            args = self._args(by)
        else:
            args = self._args.iterate(by)

        return self.mod(*args.args, **args.kwargs).val


class Streamer(object):

    def __init__(self, iterator):

        self._iterator = iterator
        self._cur = None
        self._output = UNDEFINED
        self._prev = None
        self._dx = None

    @property
    def complete(self) -> bool:
        return self._output is not UNDEFINED

    def __call__(self) -> typing.Union[typing.Any, Partial]:

        try:
            self._prev = self._cur
            self._cur, self._dx = next(self._iterator)
            return Partial(self._cur, self._prev, self._dx, False)    
        except StopIteration:
            self._output = self._cur
            return Partial(self._cur, self._prev, self._dx, True) 


class WaitSrc(Src):

    def __init__(self, incoming: T):

        super().__init__()
        self._incoming = incoming

    def incoming(self) -> typing.Iterator['T']:
        
        yield self._incoming

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        # what if "by" is partial
        if self in by:
            return by[self]
        
        if isinstance(self._incoming.val, Partial) or isinstance(self._incoming.val, Streamer):
            return WAITING
        return self._incoming.val
    

def wait(t: T) -> T:

    if isinstance(t.val, Partial) or isinstance(t.val, Streamer):
        val = WAITING
    else:
        val = t.val
    
    return T(val, WaitSrc(t))


def stream(module: 'StreamableModule', *args, interval: float=None, **kwargs) -> typing.Iterator[T]:

    if not isinstance(module, StreamableModule):
        raise RuntimeError('Stream only works for streamable modules')
    t = module(*args, **kwargs)
    yield t

    if isinstance(t.val, Streamer):
        while not t.val.complete:
            if interval is not None:
                time.sleep(interval)
            yield t


class Module(ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        pass

    def __call__(self, *args, **kwargs) -> T:

        args = Args(*args, **kwargs)
        if not args.is_undefined():
            partial = args.has_partial()
            args = args.eval()
            res = self.forward(*args.args, **args.kwargs)
            if partial:
                res = Partial(res)
            return T(
                res,
                FSrc(self, args)
            )
  
        return T(
            UNDEFINED, FSrc(self, args)
        )
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)
    
    # async def __async_call__(self, *args, **kwargs) -> T:

    #     args = Args(*args, **kwargs)
    #     if not args.undefined:
    #         partial = args.has_partial()
    #         args = args.eval()
    #         res = await self.async_forward(*args.args, **args.kwargs)
    #         if partial:
    #             res = Partial(res)
    #         return T(
    #             res,
    #             FSrc(self, args)
    #         )
  
    #     return T(
    #         UNDEFINED, FSrc(self, args)
    #     )


class StreamableModule(Module, ABC):

    @abstractmethod
    def stream_iter(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        pass 

    def forward(self, *args, **kwargs) -> Streamer:

        return Streamer(
            self.stream_iter(*args, **kwargs)
        )

    def __call__(self, *args, **kwargs) -> T:

        args = Args(*args, **kwargs)
        
        if not args.is_undefined():
            args = args.eval()
            return T(
                self.forward(*args.args, **args.kwargs),
                StreamSrc(self, args)
            )

        return T(
            UNDEFINED, StreamSrc(self, args)
        )


class ParallelModule(Module, ABC):

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    def __getitem__(self, idx: int) -> 'Module':

        return self._modules[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for module_i in self._modules:
            yield module_i

    def __call__(self, *args: Args) -> T:

        undefined = False
        has_partial = False
        for a in args:
            undefined = a.is_undefined() or undefined
            has_partial = a.has_partial() or has_partial

        if not undefined:
            
            args = [a.eval() for a in args]
            res = self.forward(*args)
            if has_partial:
                res = Partial(res)
            return T(
                res,
                ParallelSrc(self, args)
            )
  
        return T(
            UNDEFINED, ParallelSrc(self, args)
        )


class ParallelSrc(Src):

    def __init__(self, module: 'ParallelModule', args: typing.List['Args']) -> None:
        super().__init__()
        self._module = module
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        
        for arg in self._args:
            for incoming in arg.incoming():
                yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        args = [arg(by) for arg in self._args]
        return self._module(*args).val
    
    def __getitem__(self, idx) -> 'Module':

        return self._module[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for mod_i in self._module:
            yield mod_i

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        return self.forward(by)

# with stream()
#    ... # set these ones to "use partial"
#    ... # when probing if it is 
# # execute the stream afterward if possible
# # normally "use partial" is false

# complete 1) probes if there is "partial" input
# if undefined input, will not probe
# val is set to an "iterator" if the value is
# valid. if iterator "complete" will replace


# How to handle everything outside the loop?
# and konw it 
# with stream()
#    ...
# # execute the loop if not "UNDEFINED"
# # If UNDEFINED => 

# use MultiT
# ParallelSrc inherits from WrapSrc 
# 


# TODO: 
#  - Add Stream
#  - Add DocStrings
#  - Add tests
#  1) use a thread
#  2) Know when it has completed
#  3) t = stream(t)
#     
#  4) t.completed == False
#  5) t = x(t)
#  6) t.completed == False
#  7) t = complete(t)
#  8) t.completed = True
#   1) t.completed == False if not probed yet
#   2) t.completed == False if not finished yet
#   3) t.val == UNDEFINED if not 

#   t.val is automatically updated

#   return whether streaming


#   if t.streaming is False:
#      probe again
#   y = t.val 
#     1) check if copmleted, 
#     2) if not completed probe again
#   # cannot stack "streamed" unless completed

# class Graph:
    
#     def __init__(self, in_: typing.Union[typing.List[T], T], out_: typing.Union[typing.List[T], T]):

#         self._in = in_
#         self._out = out_

#     def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
#         pass


# class StreamSrc(Src):
#     # use to nest streaming operations

#     def __init__(self, graph: Graph):
        
#         super().__init__()
#         self.graph = graph

#     def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
#         return super().forward(by)
