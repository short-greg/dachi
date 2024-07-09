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
from .._core import Struct


class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'

UNDEFINED = _Types.UNDEFINED
WAITING = _Types.WAITING


def is_undefined(val) -> bool:
    """
    Args:
        val : The value to check

    Returns:
        bool: Whether the value is undefined or not
    """
    return val is UNDEFINED or val is WAITING


class Src(ABC):
    """Base class for Src. Use to specify how the
    Transmission (T) was generated
    """

    @abstractmethod
    def incoming(self) -> typing.Iterator['T']:
        pass

    @abstractmethod
    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        pass

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        return self.forward(by)


class StreamSrc(Src):
    """A source used for streaming inputs such
    as streaming from an LLM
    """

    def __init__(self, module: 'StreamableModule', args: 'Args') -> None:
        """Create a Src which will handle the streaming of inputs

        Args:
            module (StreamableModule): The module to stream
            args (Args): The arguments passed into the streamer
        """
        super().__init__()

        self._module = module
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        """Loop over all incoming transmission

        Yields:
            T: The incoming transmissions to the Src
        """
        for incoming in self._args.incoming:
            yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        """

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs into the network

        Returns:
            Streamer: the streamer used by the module
        """
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
    """Class for storing a partial output from a streaming process
    """
    cur: typing.Any
    prev: typing.Any = None
    dx: typing.Any = None
    complete: bool = False


class T(object):
    """Trans
    """

    def __init__(
        self, val=UNDEFINED, src: Src=None,
        name: str=None, annotation: str=None
    ):
        """Create a "Transmission" object to pass
        through the network.

        Args:
            val (optional): The input value. Defaults to UNDEFINED.
            src (Src, optional): The Src of the transmission. Defaults to None.
            name (str, optional): The name of the transmission. Defaults to None.
            annotation (str, optional): The annotations for the transmission. Defaults to None.
        """
        self._val = val
        self._src = src
        self.name = name
        self.annotation = annotation

    @property
    def src(self) -> 'Src':
        """
        Returns:
            Src: The Src of the transmission
        """
        return self._src

    # Move up a level
    def label(self, name: str=None, annotation: str=None) -> Self:
        """Add a label to the transmission

        Args:
            name (str, optional): The name of the transmission, if None will not update. Defaults to None.
            annotation (str, optional): The annotation for the transmission. If None will not update. Defaults to None.

        Returns:
            Self: The transmission with the label
        """
        if name is not None:
            self.name = name
        if annotation is not None:
            self.annotation = annotation
        return self

    @property
    def val(self) -> typing.Any:
        """
        Returns:
            typing.Any: The value for the transmission
        """
        return self._val

    def is_undefined(self) -> bool:
        """
        Returns:
            bool: Whether the transmission is Undefineed (WAITING or UNDEFINED)
        """
        return is_undefined(self._val)

    def __getitem__(self, idx: int) -> 'T':
        """
        Args:
            idx (int): The index to use on the transmission

        Returns:
            T: A transmission indexed.
        """
        # TODO: Add multi-indices
        if is_undefined(self._val):
            return T(
                self._val, IdxSrc(self, idx)
            )

        return T(
            self._val[idx], IdxSrc(self, idx)
        )
    
    def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        """Probe the graph using the values specified in by

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs to the network

        Raises:
            RuntimeError: If the value is undefined

        Returns:
            typing.Any: The value returned by the probe
        """
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
        
        raise RuntimeError('Val has not been defined and no source for T')

    def detach(self) -> 'T':
        """Remove the Src from T. This "detaches" T from the network

        Returns:
            T: The detached T
        """
        return T(
            self._val, None
        )


class Var(Src):
    """A Variable Src that stores a default value.
    """
    
    def __init__(self, default=None, default_factory=None):
        """A Variable Src

        Args:
            default (optional): The default value for the Src. Defaults to None.
            default_factory (optional): A factory to retrieve the default. Defaults to None.
        """
        self.default = default
        self.default_factory = default_factory
        if default is None and default_factory is None:
            raise RuntimeError('Either the default value or default factory must be defined')

    def incoming(self) -> typing.Iterator[T]:
        """
        Yields:
            Iterator[typing.Iterator[T]]: The incoming Transmissions - Since this is a variable Src it returns None
        """
        # hack to ensure it is a generator
        if False:
            yield False
        
    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        """

        Args:
            by (typing.Dict[T, typing.Any]): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            typing.Any: The value of the variable. If the 
        """
        if self.default is not None:
            return self.default
        return self.default_factory()


class IdxSrc(Src):

    def __init__(self, t: T, idx):
        """Index the 

        Args:
            t (T): The Transmission to index
            idx: The index for the output of the transmission
        """
        self.t = t
        self.idx = idx

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: The indexed transmission
        """
        yield self.t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        """
        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The indexed value
        """
        val = self.t.probe(by)
        if is_undefined(val):
            return val
        return val[self.idx]


class Args(object):

    def __init__(self, *args, **kwargs):
        """Create a storage for the arguments to a module
        Consists of two components 
         - Args (value arguments)
         - Kwargs (key value argments)
        """
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
        """
        Returns:
            bool: Whether the Args are undefined
        """
        return self._undefined
    
    def eval(self) -> Self:
        """Evaluate the current arguments
         - The value of t
         - The current value for a Streamer or Partial

        Returns:
            Self: Evaluate the 
        """
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
        """_summary_

        Returns:
            typing.List: The args component of the Args
        """
        return self._args
    
    @property
    def kwargs(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The kwargs component of the Args
        """
        return self._kwargs
    
    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
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
        
    def __call__(self, by: typing.Dict['T', typing.Any]) -> Self:
        return self.forward(by)


# use partial goes in the module

class ModSrc(Src):

    def __init__(self, mod: 'Module', args: Args):
        """Create a Src for the transmission output by a module

        Args:
            mod (Module): The module souurce
            args (Args): The args to the module
        """
        super().__init__()
        self.mod = mod
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        """Loop over all incoming transmissions to the module

        Yields:
            T: The incoming transmissions to the module
        """
        for t in self._args.incoming():
            yield t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        """Execute the module that generates the T

        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The value output by the module
        """
        
        args = self._args(by)

        return self.mod(*args.args, **args.kwargs).val


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


class WaitSrc(Src):
    """Indicates to wait until completed
    """

    def __init__(self, incoming: T):
        """

        Args:
            incoming (T): 
        """
        super().__init__()
        self._incoming = incoming

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: The incoming Transmission
        """
        yield self._incoming

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        """
        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The output of the Src
        """
        if isinstance(self._incoming.val, Partial) and not self._incoming.val.complete:
            return WAITING

        if isinstance(self._incoming.val, Streamer) and not self._incoming.val.complete:
            return WAITING
        return self._incoming.val
    

def wait(t: T) -> T:
    """Specify to wait for a streamed Transmission to complete before executing

    Args:
        t (T): The transmission

    Returns:
        T: The T to wait for the output
    """

    if isinstance(t.val, Partial) or isinstance(t.val, Streamer):
        val = WAITING
    else:
        val = t.val
    
    return T(val, WaitSrc(t))


def stream(module: 'StreamableModule', *args, interval: float=None, **kwargs) -> typing.Iterator[T]:
    """Use to loop over a streamable module until complete

    Args:
        module (StreamableModule): The module to stream over
        interval (float, optional): The interval to stream over. Defaults to None.

    Raises:
        RuntimeError: If the module is not "Streamable"

    Yields:
        Iterator[typing.Iterator[T]]: _description_
    """
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
    """Base class for Modules
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:

        return self.forward(*args, **kwargs)

    def link(self, *args, **kwargs) -> T:
        """
        Returns:
            T: The transmission output by the module
        """
        args = Args(*args, **kwargs)
        if not args.is_undefined():
            partial = args.has_partial()
            args = args.eval()
            res = self.forward(*args.args, **args.kwargs)
            if partial:
                res = Partial(res)
            return T(
                res,
                ModSrc(self, args)
            )
  
        return T(
            UNDEFINED, ModSrc(self, args)
        )
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)


class StreamableModule(Module, ABC):
    """Module that defines a "stream_iter" method which
    allows for streaming outputs
    """

    @abstractmethod
    def stream_iter(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        pass 

    def forward(self, *args, **kwargs) -> Streamer:
        """
        Returns:
            Streamer: The Streamer to loop over
        """
        return Streamer(
            self.stream_iter(*args, **kwargs)
        )

    # TDOO: add async_link?
    def link(self, *args, **kwargs) -> T:
        """
        Returns:
            T: The Streamable transmission
        """
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
    """Module that allows 
    """

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    def __getitem__(self, idx: int) -> 'Module':

        return self._modules[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for module_i in self._modules:
            yield module_i

    def link(self, args: typing.Union[typing.List[Args], Args]) -> T:
        """
        Args:
            args (typing.Union[typing.List[Args], Args]): Either a list of args or
            a singular Arg. If it is singular "arg" it will use those
            args as the inputs to all

        Returns:
            T: The parallel Transmission
        """
        undefined = False
        has_partial = False

        if isinstance(args, Args):
            undefined = args.is_undefined()
            has_partial = args.has_partial()
        else:
            for a in args:
                undefined = a.is_undefined() or undefined
                has_partial = a.has_partial() or has_partial

        if not undefined:
            
            if isinstance(args, Args):
                args = a.eval()
                args = [a] * len(self._modules)
            else:
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
    """Create a Src for processing Parallel modules
    """

    def __init__(self, module: 'ParallelModule', args: typing.Union[Args, typing.List['Args']]) -> None:
        """_summary_

        Args:
            module (ParallelModule): The Parallel module to process for
            args (typing.Union[Args, typing.List[Args]]): The arguments to use as inputs
        """
        super().__init__()
        self._module = module
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        
        args = self._args if not isinstance(self._args, Args) else [self._args]
        
        for arg in args:
            for incoming in arg.incoming():
                yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        """Execute the Parallel module

        Args:
            by (typing.Dict[T, typing.Any]): The input to the module

        Returns:
            typing.Any: _description_
        """
        if isinstance(self._args, Args):
            args = args(by)
            return self._module(args).val

        args = [arg(by) for arg in self._args]
        return self._module(args).val
    
    def __getitem__(self, idx) -> 'Module':
        """
        Args:
            idx: The index to get the module for

        Returns:
            Module: The module specified by the index
        """
        return self._module[idx]

    def __iter__(self) -> typing.Iterator['Module']:
        """
        Yields:
            Module: The modules comprising the parallel module
        """
        for mod_i in self._module:
            yield mod_i

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        """

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): 

        Returns:
            typing.Any: 
        """
        return self.forward(by)


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

    def forward(self, *args, **kwargs) -> typing.Any:
        if self.instance:
            return self.f(self.instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def link(self, *args, **kwargs) -> typing.Any:
        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):
        # Bind the method to the instance
        return _DecMethod(self.f, instance)


def process(f):
    return _DecMethod(f)
