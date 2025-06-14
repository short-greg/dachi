
# 1st party
from abc import abstractmethod, ABC
import typing
import time
from typing import Self

# local
from . import (
    Partial, Streamer,
    Module, StreamModule
)
from ..utils import is_undefined
from ..utils import UNDEFINED, WAITING


class T(object):
    """
    """
    def __init__(
        self, val=UNDEFINED, src: 'Src'=None,
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
    
    def probe(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
        """Probe the graph using the values specified in by

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs to the network

        Raises:
            RuntimeError: If the value is undefined

        Returns:
            typing.Any: The value returned by the probe
        """
        by = by or {}
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


class Src(Module):
    """Base class for Src. Use to specify how the
    Transmission (T) was generated
    """

    @abstractmethod
    def incoming(self) -> typing.Iterator['T']:
        pass

    @abstractmethod
    def forward(
        self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
        pass

    # def forward(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
    #     return self.forward(by)


class StreamSrc(Src):
    """A source used for streaming inputs such
    as streaming from an LLM
    """
    def __init__(self, module: 'Module', args: 'NodeArgs') -> None:
        """Create a Src which will handle the streaming of inputs

        Args:
            module (Module): The module to stream
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
        for incoming in self._args.incoming():
            yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
        """

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs into the network

        Returns:
            Streamer: the streamer used by the module
        """
        by = by or {}
        if self in by:
            value: Streamer = by[self]
            return value
    
        args = self._args(by)
        streamer = Streamer(
            self._module.stream(
                *args.args,
                **args.kwargs
            )
        )

        return streamer
    
    def __call__(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
        
        return self.forward(by)


class NodeArgs(object):
    """
    """

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
                
        if not undefined:
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
                a = a().dx
            if isinstance(a, Partial):
                a = a.dx
            args.append(a)
        
        for k, a in self._kwargs.items():
            if isinstance(a, T):
                a = a.val
            if isinstance(a, Streamer):
                a = a()
            if isinstance(a, Partial):
                a = a.dx
            kwargs[k] = a
        return NodeArgs(*args, **kwargs)
    
    @property
    def args(self) -> typing.List:
        """Get the args

        Returns:
            typing.List: The args component of the Args
        """
        return self._args
    
    @property
    def kwargs(self) -> typing.Dict:
        """Get the kwargs
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
            elif isinstance(arg, Var):
                yield arg

        for k, arg in self._kwargs:
            if isinstance(arg, T):
                yield arg
            elif isinstance(arg, Var):
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
    
    def forward(
        self, 
        by: typing.Dict['T', typing.Any]=None
    ) -> Self:

        by = by or {}
        args = []
        kwargs = {}
        for arg in self._args:
            is_t = isinstance(arg, T)
            
            if is_t and arg in by:
                val = by[arg]
                if isinstance(val, Partial):
                    # partial = True
                    args.append(val.cur)

                else:
                    args.append(val)
            elif is_t and arg.val is not UNDEFINED:
                args.append(arg.val)
            elif isinstance(arg, Var):
                args.append(arg(by))
            elif is_t:
                raise ValueError(f'Arg has not been defined')  
            else:
                args.append(arg)
            
        for k, arg in self._kwargs.items():
            is_t = isinstance(arg, T)
            if is_t and arg in by:
                kwargs[k] = by[arg]
            elif is_t and arg.val is not UNDEFINED:
                kwargs[k] = arg.val
            elif is_t:
                raise ValueError(f'Arg {k} has not been defined')  
            else:
                kwargs[k] = arg
        
        return NodeArgs(*args, **kwargs)
        
    def __call__(self, by: typing.Dict['T', typing.Any]=None) -> Self:
        return self.forward(by)


class ModSrc(Src):

    def __init__(self, mod: 'Module', args: NodeArgs=None):
        """Create a Src for the transmission output by a module

        Args:
            mod (Module): The module souurce
            args (Args): The args to the module
        """
        super().__init__()
        self.mod = mod
        if args is None:
            args = tuple()
        if not isinstance(args, NodeArgs):
            args = NodeArgs(*args)
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        """Loop over all incoming transmissions to the module

        Yields:
            T: The incoming transmissions to the module
        """
        for t in self._args.incoming():
            yield t

    def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
        """Execute the module that generates the T

        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The value output by the module
        """
        by = by or {}
        args = self._args(by)
        return link(self.mod, *args.args, **args.kwargs).val
    
    @classmethod
    def create(cls, mod: 'Module', *args, **kwargs) -> Self:
        """Classmethod to create a ModSrc with args and kwargs

        Args:
            mod (Module): The module

        Returns:
            Self: The ModSrc
        """
        return cls(
            mod, NodeArgs(*args, **kwargs)
        )


def wait(
    t: T, 
    agg: typing.Callable[[typing.Any], typing.Any]=None
) -> T:
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
    
    return T(val, WaitSrc(t, agg))


class WaitSrc(Src):
    """Indicates to wait until completed
    """
    def __init__(
        self, incoming: T, 
        agg: typing.Callable[[typing.Any], typing.Any]=None
    ):
        """Create a Src to wait for the incoming transmission

        Args:
            incoming (T): The incoming transmission
        """
        super().__init__()
        self._incoming = incoming
        self._agg = agg or (lambda x: x)

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: The incoming Transmission
        """
        yield self._incoming

    def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
        """
        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The output of the Src
        """
        by = by or {}
        
        val = self._incoming.probe(by)

        if isinstance(val, Partial) and not val.complete:
            return WAITING
        if isinstance(val, Partial):
            return self._agg(val.full)

        if isinstance(val, Streamer):
        
            if not val.complete:
                res = val()
                if res.complete:
                    return self._agg(res.full)
                return WAITING

            return self._agg(val.output.full)
            
        return val


class Var(Src):
    """A Variable Src that stores a default value.
    """
    
    def __init__(self, default=UNDEFINED, default_factory=UNDEFINED):
        """A Variable Src

        Args:
            default (optional): The default value for the Src. Defaults to None.
            default_factory (optional): A factory to retrieve the default. Defaults to None.
        """
        self.default = default
        self.default_factory = default_factory
        if default is UNDEFINED and default_factory is UNDEFINED:
            raise RuntimeError('Either the default value or default factory must be defined')

    def incoming(self) -> typing.Iterator[T]:
        """
        Yields:
            Iterator[typing.Iterator[T]]: The incoming Transmissions - Since this is a variable Src it returns None
        """
        # hack to ensure it is a generator
        if False:
            yield False
        
    def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
        """Return the value of the variable

        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Raises:
            RuntimeError: If the default value is not defined

        Returns:
            typing.Any: The value of the variable. If the 
        """
        by = by or {}
        if self.default is not UNDEFINED:
            return self.default
        if self.default_factory is not UNDEFINED:
            return self.default_factory()
        raise RuntimeError('There is no default for var.')


class IdxSrc(Src):
    """Index the output of a transmission"""

    def __init__(self, t: T, idx):
        """Index the output of a transmission

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

    def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
        """
        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The indexed value
        """
        by = by or {}
        val = self.t.probe(by)
        if is_undefined(val):
            return val
        return val[self.idx]


def link(module: Module, *args, **kwargs) -> T:
    """Create a link to a module

    Args:
        module (Module): The module to link to
        *args: The arguments to the module
        **kwargs: The keyword arguments to the module
    """
    args = NodeArgs(*args, **kwargs)
    if not args.is_undefined():
        partial = args.has_partial()
        args = args.eval()
        res = module.forward(*args.args, **args.kwargs)
        if partial:
            res = Partial(res)
        return T(
            res,
            ModSrc(module, args)
        )

    return T(
        UNDEFINED, ModSrc(module, args)
    )


# TODO: Combine the following two
# TDOO: add async_link?
# streamable
def stream_link(module: 'StreamModule', *args, **kwargs) -> T:
    """
    Returns:
        T: The Streamable transmission
    """
    args = NodeArgs(*args, **kwargs)
    
    if not args.is_undefined():
        args = args.eval()

        # cur = module.stream(*args.args, **args.kwargs)
        return T(
            Streamer(module.stream(*args.args, **args.kwargs)),
            StreamSrc(module, args)
        )

    return T(
        UNDEFINED, StreamSrc(module, args)
    )


def stream(module: 'StreamModule', *args, interval: float=None, **kwargs) -> typing.Iterator[T]:
    """Use to loop over a streamable module until complete

    Args:
        module (Module): The module to stream over
        interval (float, optional): The interval to stream over. Defaults to None.

    Raises:
        RuntimeError: If the module is not "Streamable"

    Yields:
        Iterator[typing.Iterator[T]]: _description_
    """
    if not isinstance(module, Module):
        raise RuntimeError('Stream only works for streamable modules')
    t = stream_link(module, *args, **kwargs)
    yield t

    if isinstance(t.val, Streamer):
        while not t.val.complete:
            if interval is not None:
                time.sleep(interval)
            yield t
