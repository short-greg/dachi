# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self

# 3rd party
import networkx as nx
import functools
import time

from dataclasses import dataclass
import uuid


class _UndefinedType:
    def __repr__(self):
        return 'UNDEFINED'

# Create a singleton instance of UndefinedType
UNDEFINED = _UndefinedType()


class _WaitingType:
    def __repr__(self):
        return 'WAITING'

# Create a singleton instance of UndefinedType
WAITING = _WaitingType()


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
            return value()
    
        args = self._args.iterate(by)        
        streamer = by[self] = self._module.stream(*args.args, **args.kwargs)
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
        multi: bool=False, use_partial: bool=False, name: str=None, annotation: str=None
    ):
        self._val = val
        self._src = src
        self._multi = multi
        self._name = name
        self._annnotation = annotation
        self._use_partial = use_partial

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

    @property
    def undefined(self) -> bool:

        return self._val is UNDEFINED

    def __getitem__(self, idx: int) -> 'T':
        
        if not self._multi:
            raise RuntimeError(
                'Object T does not have multiple objects'
            )
        else:
            val = None
        return T(
            val, IdxSrc(self, idx)
        )
    
    def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        if self._val is not UNDEFINED:
            return self._val

        if self in by and not isinstance(by[self], Partial):
            return by[self]
    
        if self._src is not None:
            for incoming in self._src.incoming():
                incoming.probe(by)
            by[self] = self.src(by)
            return by[self]
        
        raise RuntimeError('Val has not been defined and no source for t')

    def detach(self):
        return T(
            self._val, None, self._multi
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
        if val is UNDEFINED:
            return val
        return val[self.idx]


class Args(object):

    def __init__(self, *args, **kwargs):
        
        undefined = False

        for arg in args:
            
            if isinstance(arg, T):
                if arg.undefined:
                    undefined = True
                    break
        for k, arg in kwargs.items():
            
            if isinstance(arg, T):
                if arg.undefined:
                    undefined = True
                    break
        self._args = args
        self._undefined = undefined
        self._kwargs = kwargs
    
    @property
    def undefined(self) -> bool:
        return self._undefined
    
    def eval(self) -> Self:

        if self._undefined:
            return None
        args = [a.val if isinstance(a, T) else a for a in self._args]
        kwargs = {k: a.val if isinstance(a, T) else a for k, a in self._kwargs.items()}
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
            if isinstance(a, Partial) and not a.complete:
                return True
        for k, a in self._kwargs.items():
            if isinstance(a, Partial) and not a.complete:
                return True
        return False
    
    def forward(self, by: typing.Dict['T', typing.Any]=None) -> Self:

        by = by or {}
        args = []
        kwargs = {}
        # partial = False
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

    def __init__(self, mod: 'Module', args: Args, stream_partial: bool=False):

        super().__init__()
        self.mod = mod
        self._args = args
        self._stream_partial = stream_partial

    def incoming(self) -> typing.Iterator['T']:
        
        for t in self._args.incoming():
            yield t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        # what if "by" is partial
        if self in by:
            return by[self]
        
        if self._stream_partial or not self._args.has_partial():
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

    def __call__(self) -> typing.Union[typing.Any, Partial]:

        try:
            self._prev = self._cur
            self._cur, self._dx = next(self._iterator)
            return Partial(self._cur, self._prev, self._dx, False)    
        except StopIteration:
            self._output = self._cur
            return Partial(self._cur, self._prev, self._dx, True) 


def stream(module: 'Module', *args, **kwargs) -> 'T':

    return module(*args, stream_partial=True, **kwargs)


class Module(ABC):

    def __init__(self, multi_out: bool=False, stream_partial: bool=False):
        """

        Args:
            multi_out (bool, optional): . Defaults to False.
            stream_partial (bool, optional): . Defaults to False.
        """
        self._multi_out = multi_out
        self._stream_partial = stream_partial

    @property
    def stream_partial(self) -> bool:

        return self._stream_partial
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        pass

    def __call__(self, *args, stream_partial: bool=False, **kwargs) -> T:

        args = Args(*args, **kwargs)
        if not args.undefined:
            args = args.eval()
            return T(
                self.forward(*args.args, **args.kwargs),
                FSrc(self, args, stream_partial=stream_partial), self._multi_out
            )
        
        return T(
            UNDEFINED, FSrc(self, args, stream_partial=stream_partial), self._multi_out
        )
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)
    
    async def __async_call__(self, *args, **kwargs) -> T:

        return self(*args, **kwargs)


class StreamableModule(Module, ABC):

    @abstractmethod
    def stream_iter(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        pass 

    # def stream(self, *args, **kwargs) -> 'T':
        
    #     return T(
    #         UNDEFINED, StreamSrc(self, Args(*args, **kwargs)),
    #         self._multi_out
    #     )

    def forward(self, *args, **kwargs) -> Streamer:

        return Streamer(
            self.stream_iter(*args, **kwargs)
        )

    def __call__(self, *args, **kwargs) -> T:

        args = Args(*args, **kwargs)
        
        if not args.undefined:
            args = args.eval()
            return T(
                self.forward(*args.args, **args.kwargs),
                StreamSrc(self, args), self._multi_out
            )

        return T(
            None, StreamSrc(self, args), self._multi_out
        )


class WrapIdxSrc(Src):

    def __init__(self, t: 'MultiT', idx):

        self.t = t
        self.idx = idx

    def incoming(self) -> typing.Iterator['T']:
        yield self.t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by[self]
        val = self.t.probe(by)
        if val is UNDEFINED:
            return val
        return val[self.idx]


class WrapSrc(ABC):

    @abstractmethod
    def incoming(self) -> typing.Iterator['T']:
        pass

    @abstractmethod
    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        pass

    @abstractmethod
    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        pass


class MultiT(T):

    def __init__(self, vals: typing.Tuple, src: 'WrapSrc'):
        
        self._vals = vals
        self._src = src

    @property
    def undefined(self) -> bool:

        return functools.reduce(lambda x, y: x and y.undefined, self._ts)

    def __getitem__(self, idx: int) -> 'T':
        
        return T(
            self._vals[idx], WrapIdxSrc(self, idx),
            self._src[idx].multi
        )

    def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        if self._vals is not None:
            return self._vals

        if self in by:
            return by[self]
    
        if self._src is not None:
            for incoming in self._src.incoming():
                incoming.probe(by)
            by[self] = self.src(by)
            return by[self]
        
        raise RuntimeError('Val has not been defined and no source for t')

    def detach(self):
        return MultiT(
            self._vals, None
        )



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
