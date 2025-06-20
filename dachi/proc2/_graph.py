
# 1st party
from abc import abstractmethod, ABC
import typing
import time
from typing import Self
import asyncio

# local
from ..utils import is_undefined
from ..utils import UNDEFINED, WAITING
from ._process import Process, AsyncProcess
from ..core import BaseModule
from ..core import SerialDict

from ._process import Partial
from ._process import StreamProcess, AsyncStreamProcess

from ..utils import is_async_function

from dataclasses import InitVar


# TODO: Check if the value coming from incoming is undefined or waiting... 

class BaseNode(AsyncProcess):
    """
    """

    name: str | None
    val: typing.Any = UNDEFINED
    annotation: str | None = None
    
    # def incoming(self) -> typing.Iterator['BaseNode']:
    #     """
    #     Yields:
    #         T: The indexed transmission
    #     """
    #     pass

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

    def __getitem__(self, idx: int) -> 'BaseNode':
        """
        Args:
            idx (int): The index to use on the transmission

        Returns:
            T: A transmission indexed.
        """
        # TODO: Add multi-indices
        if is_undefined(self._val):
            return T(
                self._val, Idx(self, idx)
            )

        return T(
            self._val[idx], Idx(self, idx)
        )
    
    def detach(self) -> typing.Self:
        """Remove the Src from T. This "detaches" T from the network

        Returns:
            T: The detached T
        """
        return T(
            self._val, None
        )
    
    async def aforward(
        self, by: typing.Dict['T', typing.Any]=None
    ) -> typing.Any:
        pass

    def incoming(self) -> typing.Iterator['BaseNode']:
        pass


class Var(BaseNode):
    """
    """
    async def aforward(
        self, by: typing.Dict['T', typing.Any]=None
    ) -> typing.Any:
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

        if (
            self in by 
            # and not isinstance(by[self], Streamer) 
            and not isinstance(by[self], Partial)
        ):
            return by[self]
    
        if self._src is not None:
            for incoming in self._src.incoming():
                incoming.probe(by)
            val = by[self] = self.src(by)
            # if isinstance(val, Streamer):
            #     val = val()
            return val
        
        raise RuntimeError('Val has not been defined and no source for T')
    
    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        
        if False:
            yield None
        return


class ProcNode(BaseNode):

    args: SerialDict

    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self._kwargs:
            if isinstance(arg, BaseNode):
                yield k, arg

    def has_partial(self) -> bool:

        for k, a in self._kwargs.items():
            if isinstance(a, T):
                a = a.val
            if (isinstance(a, Partial) and not a.complete):
                return True
        return False

    def eval_args(self) -> SerialDict:
        """Evaluate the current arguments
            - The value of t
            - The current value for a Streamer or Partial

        Returns:
            Self: Evaluate the 
        """
        if self._undefined:
            return None
        kwargs = {}
        
        for k, a in self.args.items():
            if isinstance(a, 'BaseNode'):
                a = a.val
            elif isinstance(a, Partial):
                a = a.dx
            kwargs[k] = a
        return SerialDict(items=kwargs)

    # Have to evaluate the kwargs    
    async def get_incoming(
        self, 
        by: typing.Dict['T', typing.Any]=None
    ) -> Self:

        by = by or {}
        kwargs = {}
        tasks = {}
        with asyncio.TaskGroup() as tg:

            for k, arg in self.args.items():
                is_t = isinstance(arg, BaseNode)
                if is_t and arg in by:
                    kwargs[k] = by[arg]
                elif is_t and arg.val is not UNDEFINED:
                    kwargs[k] = arg.val
                elif is_t:
                    tasks[k] = tg.create_task(
                        arg.aforward(by)
                    )
                else:
                    kwargs[k] = arg
        for k, task in tasks.items():
            kwargs[k] = task.result()
        
        return SerialDict(kwargs)

    def has_partial(self) -> bool:

        for k, a in self._kwargs.items():
            if isinstance(a, T):
                a = a.val
            if (isinstance(a, Partial) and not a.complete):
                return True
        return False

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self._kwargs:
            if isinstance(arg, BaseNode):
                yield arg


class T(ProcNode):
    """...
    """
    src: Process | AsyncProcess
    is_async: bool = False

    def __post_init__(self, args: SerialDict):

        self._args = args
            
    async def aforward(
        self, by: typing.Dict['BaseNode', typing.Any]=None
    ) -> typing.Any:
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

        if self in by:
            return by[self]
            
        args, kwargs = self.args(by)
        
        if self._is_async:
            val = by[self] = await self.src(*args, **kwargs)
        else:
            val = by[self] = self.src(*args, **kwargs)    

        if val is UNDEFINED:
            raise RuntimeError(
                'Val has not been defined and no source for T'
            )
        return val


class Stream(BaseNode):
    """...
    """
    src: StreamProcess | AsyncStreamProcess
    args: SerialDict
    is_async: bool = False

    def __post_init__(self):

        self._is_async = is_async_function(self.src)
        self._stream = None
        self._prev = None
        self._full = []
            
    async def aforward(
        self, by: typing.Dict['BaseNode', typing.Any]=None
    ) -> typing.Any:
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
        
        if self._stream is None:

            args, kwargs = self.args(by)
        
            if self._is_async:
                self._stream = aiter(self.src(*args, **kwargs))
            else:
                self._stream = iter(self.src(*args, **kwargs))

        try:
            if self._is_async:
                self._dx = await anext(self._stream)
            else:
                self._dx = next(self._stream)
            self._full.append(self._dx)
            self._val = by[self] = Partial(
                dx=self._dx, complete=False, prev=self._prev,
                full=self._full
            )
        except StopIteration:
            self._val = by[self] = Partial(
                dx=None, complete=True,
                prev=self._dx, full=self._full
            )

        return self._val


def t(
    p: Process, 
    _name: str=None, _annotation: str=None,
    **kwargs, 
) -> T:

    args = SerialDict(kwargs)
    return T(
        src=p, args=args, name=_name, annotation=_annotation,
        is_async=False
    )


def async_t(
    p: AsyncProcess,
    _name: str=None, _annotation: str=None,
    **kwargs, 
) -> T:

    args = SerialDict(kwargs)
    return T(
        src=p, args=args, name=_name, annotation=_annotation,
        is_async=True
    )
    

def stream(
    p: StreamProcess | AsyncStreamProcess, 
    _name: str=None, _annotation: str=None,
    **kwargs
) -> Stream:

    args = SerialDict(args, kwargs)
    return Stream(
        src=p, args=args, name=_name,
        annotation=_annotation,
        is_async=False
    )


def async_stream(
    p: StreamProcess | AsyncStreamProcess, 
    _name: str=None, _annotation: str=None,
    **kwargs
) -> Stream:

    args = SerialDict(args, kwargs)
    return Stream(
        src=p, args=args, name=_name,
        annotation=_annotation,
        is_async=True
    )


class Idx(Process):
    """Index the output of a transmission"""

    idx: int | typing.List[int]

    def index(self, val) -> typing.Union[typing.Any, typing.List[typing.Any]]:
        
        if isinstance(self.idx, typing.List):
            return [
                val[i] for i in self.idx
            ]
        return val[self.idx]

    def forward(
        self, val
    ) -> typing.Any:
        """Probe the graph using the values specified in by

        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs to the network

        Raises:
            RuntimeError: If the value is undefined

        Returns:
            typing.Any: The value returned by the probe
        """
        if isinstance(self.idx, typing.List):
            return [
                val[i] for i in self.idx
            ]
        return val[self.idx]


class WaitProcess(Process):
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

    def forward(self, val=None) -> typing.Any:
        """
        Args:
            by (typing.Dict[T, typing.Any]): The input to the network

        Returns:
            typing.Any: The output of the Src
        """
        if isinstance(val, Partial) and not val.complete:
            return WAITING
        
        return self._agg(val.full)


class DAG(Process):

    pass



# def stream(module: 'StreamModule', *args, interval: float=None, **kwargs) -> typing.Iterator[T]:
#     """Use to loop over a streamable module until complete

#     Args:
#         module (Module): The module to stream over
#         interval (float, optional): The interval to stream over. Defaults to None.

#     Raises:
#         RuntimeError: If the module is not "Streamable"

#     Yields:
#         Iterator[typing.Iterator[T]]: _description_
#     """
#     if not isinstance(module, Module):
#         raise RuntimeError('Stream only works for streamable modules')
#     t = stream_link(module, *args, **kwargs)
#     yield t

#     if isinstance(t.val, Streamer):
#         while not t.val.complete:
#             if interval is not None:
#                 time.sleep(interval)
#             yield t



# How to serialize this?
# I think basically i can 
# have everything eith


# y = t(p, x=1, y=3)
# ... What if p is a function

# 1) T.. T must be serializable (?)
# 2) T must be replaceable (?), use a reference instead
# 1) T is serializable (?)
#  


# module dict
# serializable dict # works with anything serializable, process 


    

# class TArgs(object):
#     """
#     """

#     kwargs: SerialDict

#     def __post_init__(self, **kwargs):
#         """Create a storage for the arguments to a module
#         Consists of two components 
#          - Args (value arguments)
#          - Kwargs (key value argments)
#         """
#         undefined = False

#         for arg in args:
#             if isinstance(arg, T):
#                 if is_undefined(arg.val):
#                     undefined = True
#                     break
                
#         if not undefined:
#             for k, arg in kwargs.items():
                
#                 if isinstance(arg, T):
#                     if is_undefined(arg.val):
#                         undefined = True
#                         break

#         self._undefined = undefined
    
#     def is_undefined(self) -> bool:
#         """
#         Returns:
#             bool: Whether the Args are undefined
#         """
#         return self._undefined
    
#     def eval(self) -> Self:
#         """Evaluate the current arguments
#          - The value of t
#          - The current value for a Streamer or Partial

#         Returns:
#             Self: Evaluate the 
#         """
#         if self._undefined:
#             return None
#         kwargs = {}
        
#         for k, a in self._kwargs.items():
#             if isinstance(a, 'BaseNode'):
#                 a = a.val
#             elif isinstance(a, Partial):
#                 a = a.dx
#             kwargs[k] = a
#         return TArgs(kwargs)
    
#     @property
#     def kwargs(self) -> typing.Dict:
#         """Get the kwargs
#         Returns:
#             typing.Dict: The kwargs component of the Args
#         """
#         return self._kwargs
    
#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: All of the arguments connected to another
#             transmission
#         """
#         for k, arg in self._kwargs:
#             if isinstance(arg, T):
#                 yield arg
#             elif isinstance(arg, Var):
#                 yield arg

#     def has_partial(self) -> bool:

#         for k, a in self._kwargs.items():
#             if isinstance(a, T):
#                 a = a.val
#             if (isinstance(a, Partial) and not a.complete):
#                 return True
#         return False
    
#     def forward(
#         self, 
#         by: typing.Dict['T', typing.Any]=None
#     ) -> Self:

#         by = by or {}
#         kwargs = {}
#         for k, arg in self._kwargs.items():
#             is_t = isinstance(arg, T)
#             if is_t and arg in by:
#                 kwargs[k] = by[arg]
#             elif is_t and arg.val is not UNDEFINED:
#                 kwargs[k] = arg.val
#             elif is_t:
#                 raise ValueError(f'Arg {k} has not been defined')  
#             else:
#                 kwargs[k] = arg
        
#         return TArgs(**kwargs)
        
#     def __call__(self, by: typing.Dict['T', typing.Any]=None) -> Self:
#         return self.forward(by)


# class T(BaseNode):
#     """...
#     """
#     src: Process | AsyncProcess = None
#     incoming: typing.List['BaseNode'] | None = None

#     async def aforward(
#         self, by: typing.Dict['T', typing.Any]=None
#     ) -> typing.Any:
#         """Probe the graph using the values specified in by

#         Args:
#             by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs to the network

#         Raises:
#             RuntimeError: If the value is undefined

#         Returns:
#             typing.Any: The value returned by the probe
#         """
#         by = by or {}
#         if not is_undefined(self._val):
#             return self._val

#         if self in by and not isinstance(by[self], Streamer) and not isinstance(by[self], Partial):
#             return by[self]
    
#         if self._src is not None:
#             for incoming in self._src.incoming():
#                 incoming.probe(by)
#             val = by[self] = self.src(by)
#             if isinstance(val, Streamer):
#                 val = val()
#             return val
        
#         raise RuntimeError('Val has not been defined and no source for T')


# class StreamSrc(Src):
#     """A source used for streaming inputs such
#     as streaming from an LLM
#     """
#     def __init__(self, module: 'Module', args: 'NodeArgs') -> None:
#         """Create a Src which will handle the streaming of inputs

#         Args:
#             module (Module): The module to stream
#             args (Args): The arguments passed into the streamer
#         """
#         super().__init__()
#         self._module = module
#         self._args = args

#     def incoming(self) -> typing.Iterator['T']:
#         """Loop over all incoming transmission

#         Yields:
#             T: The incoming transmissions to the Src
#         """
#         for incoming in self._args.incoming():
#             yield incoming

#     def forward(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
#         """

#         Args:
#             by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs into the network

#         Returns:
#             Streamer: the streamer used by the module
#         """
#         by = by or {}
#         if self in by:
#             value: Streamer = by[self]
#             return value
    
#         args = self._args(by)
#         streamer = Streamer(
#             self._module.stream(
#                 *args.args,
#                 **args.kwargs
#             )
#         )

#         return streamer
    
#     def __call__(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
        
#         return self.forward(by)


# class NodeArgs(object):
#     """
#     """

#     def __init__(self, *args, **kwargs):
#         """Create a storage for the arguments to a module
#         Consists of two components 
#          - Args (value arguments)
#          - Kwargs (key value argments)
#         """
#         undefined = False

#         for arg in args:
#             if isinstance(arg, T):
#                 if is_undefined(arg.val):
#                     undefined = True
#                     break
                
#         if not undefined:
#             for k, arg in kwargs.items():
                
#                 if isinstance(arg, T):
#                     if is_undefined(arg.val):
#                         undefined = True
#                         break

#         self._args = args
#         self._undefined = undefined
#         self._kwargs = kwargs
    
#     def is_undefined(self) -> bool:
#         """
#         Returns:
#             bool: Whether the Args are undefined
#         """
#         return self._undefined
    
#     def eval(self) -> Self:
#         """Evaluate the current arguments
#          - The value of t
#          - The current value for a Streamer or Partial

#         Returns:
#             Self: Evaluate the 
#         """
#         if self._undefined:
#             return None
#         args = []
#         kwargs = {}
#         for a in self._args:
#             if isinstance(a, T):
#                 a = a.val
#             if isinstance(a, Streamer):
#                 a = a().dx
#             if isinstance(a, Partial):
#                 a = a.dx
#             args.append(a)
        
#         for k, a in self._kwargs.items():
#             if isinstance(a, T):
#                 a = a.val
#             if isinstance(a, Streamer):
#                 a = a()
#             if isinstance(a, Partial):
#                 a = a.dx
#             kwargs[k] = a
#         return NodeArgs(*args, **kwargs)
    
#     @property
#     def args(self) -> typing.List:
#         """Get the args

#         Returns:
#             typing.List: The args component of the Args
#         """
#         return self._args
    
#     @property
#     def kwargs(self) -> typing.Dict:
#         """Get the kwargs
#         Returns:
#             typing.Dict: The kwargs component of the Args
#         """
#         return self._kwargs
    
#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: All of the arguments connected to another
#             transmission
#         """
#         for arg in self._args:
#             if isinstance(arg, T):
#                 yield arg
#             elif isinstance(arg, Var):
#                 yield arg

#         for k, arg in self._kwargs:
#             if isinstance(arg, T):
#                 yield arg
#             elif isinstance(arg, Var):
#                 yield arg

#     def has_partial(self) -> bool:

#         for a in self._args:
#             if isinstance(a, T):
#                 a = a.val
#             if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
                
#                 return True
#         for k, a in self._kwargs.items():
#             if isinstance(a, T):
#                 a = a.val
#             if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
#                 return True
#         return False
    
#     def forward(
#         self, 
#         by: typing.Dict['T', typing.Any]=None
#     ) -> Self:

#         by = by or {}
#         args = []
#         kwargs = {}
#         for arg in self._args:
#             is_t = isinstance(arg, T)
            
#             if is_t and arg in by:
#                 val = by[arg]
#                 if isinstance(val, Partial):
#                     # partial = True
#                     args.append(val.cur)

#                 else:
#                     args.append(val)
#             elif is_t and arg.val is not UNDEFINED:
#                 args.append(arg.val)
#             elif isinstance(arg, Var):
#                 args.append(arg(by))
#             elif is_t:
#                 raise ValueError(f'Arg has not been defined')  
#             else:
#                 args.append(arg)
            
#         for k, arg in self._kwargs.items():
#             is_t = isinstance(arg, T)
#             if is_t and arg in by:
#                 kwargs[k] = by[arg]
#             elif is_t and arg.val is not UNDEFINED:
#                 kwargs[k] = arg.val
#             elif is_t:
#                 raise ValueError(f'Arg {k} has not been defined')  
#             else:
#                 kwargs[k] = arg
        
#         return NodeArgs(*args, **kwargs)
        
#     def __call__(self, by: typing.Dict['T', typing.Any]=None) -> Self:
#         return self.forward(by)



# class NodeArgs(object):
#     """
#     """

#     def __init__(self, *args, **kwargs):
#         """Create a storage for the arguments to a module
#         Consists of two components 
#          - Args (value arguments)
#          - Kwargs (key value argments)
#         """
#         undefined = False

#         for arg in args:
#             if isinstance(arg, T):
#                 if is_undefined(arg.val):
#                     undefined = True
#                     break
                
#         if not undefined:
#             for k, arg in kwargs.items():
                
#                 if isinstance(arg, T):
#                     if is_undefined(arg.val):
#                         undefined = True
#                         break

#         self._args = args
#         self._undefined = undefined
#         self._kwargs = kwargs
    
#     def is_undefined(self) -> bool:
#         """
#         Returns:
#             bool: Whether the Args are undefined
#         """
#         return self._undefined
    
#     def eval(self) -> Self:
#         """Evaluate the current arguments
#          - The value of t
#          - The current value for a Streamer or Partial

#         Returns:
#             Self: Evaluate the 
#         """
#         if self._undefined:
#             return None
#         args = []
#         kwargs = {}
#         for a in self._args:
#             if isinstance(a, T):
#                 a = a.val
#             if isinstance(a, Streamer):
#                 a = a().dx
#             if isinstance(a, Partial):
#                 a = a.dx
#             args.append(a)
        
#         for k, a in self._kwargs.items():
#             if isinstance(a, T):
#                 a = a.val
#             if isinstance(a, Streamer):
#                 a = a()
#             if isinstance(a, Partial):
#                 a = a.dx
#             kwargs[k] = a
#         return NodeArgs(*args, **kwargs)
    
#     @property
#     def args(self) -> typing.List:
#         """Get the args

#         Returns:
#             typing.List: The args component of the Args
#         """
#         return self._args
    
#     @property
#     def kwargs(self) -> typing.Dict:
#         """Get the kwargs
#         Returns:
#             typing.Dict: The kwargs component of the Args
#         """
#         return self._kwargs
    
#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: All of the arguments connected to another
#             transmission
#         """
#         for arg in self._args:
#             if isinstance(arg, T):
#                 yield arg
#             elif isinstance(arg, Var):
#                 yield arg

#         for k, arg in self._kwargs:
#             if isinstance(arg, T):
#                 yield arg
#             elif isinstance(arg, Var):
#                 yield arg

#     def has_partial(self) -> bool:

#         for a in self._args:
#             if isinstance(a, T):
#                 a = a.val
#             if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
                
#                 return True
#         for k, a in self._kwargs.items():
#             if isinstance(a, T):
#                 a = a.val
#             if (isinstance(a, Partial) or isinstance(a, Streamer) and not a.complete):
#                 return True
#         return False
    
#     def forward(
#         self, 
#         by: typing.Dict['T', typing.Any]=None
#     ) -> Self:

#         by = by or {}
#         args = []
#         kwargs = {}
#         for arg in self._args:
#             is_t = isinstance(arg, T)
            
#             if is_t and arg in by:
#                 val = by[arg]
#                 if isinstance(val, Partial):
#                     # partial = True
#                     args.append(val.cur)

#                 else:
#                     args.append(val)
#             elif is_t and arg.val is not UNDEFINED:
#                 args.append(arg.val)
#             elif isinstance(arg, Var):
#                 args.append(arg(by))
#             elif is_t:
#                 raise ValueError(f'Arg has not been defined')  
#             else:
#                 args.append(arg)
            
#         for k, arg in self._kwargs.items():
#             is_t = isinstance(arg, T)
#             if is_t and arg in by:
#                 kwargs[k] = by[arg]
#             elif is_t and arg.val is not UNDEFINED:
#                 kwargs[k] = arg.val
#             elif is_t:
#                 raise ValueError(f'Arg {k} has not been defined')  
#             else:
#                 kwargs[k] = arg
        
#         return NodeArgs(*args, **kwargs)
        
#     def __call__(self, by: typing.Dict['T', typing.Any]=None) -> Self:
#         return self.forward(by)

# class ModSrc(Src):

#     def __init__(self, mod: 'Module', args: NodeArgs=None):
#         """Create a Src for the transmission output by a module

#         Args:
#             mod (Module): The module souurce
#             args (Args): The args to the module
#         """
#         super().__init__()
#         self.mod = mod
#         if args is None:
#             args = tuple()
#         if not isinstance(args, NodeArgs):
#             args = NodeArgs(*args)
#         self._args = args

#     def incoming(self) -> typing.Iterator['T']:
#         """Loop over all incoming transmissions to the module

#         Yields:
#             T: The incoming transmissions to the module
#         """
#         for t in self._args.incoming():
#             yield t

#     def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
#         """Execute the module that generates the T

#         Args:
#             by (typing.Dict[T, typing.Any]): The input to the network

#         Returns:
#             typing.Any: The value output by the module
#         """
#         by = by or {}
#         args = self._args(by)
#         return link(self.mod, *args.args, **args.kwargs).val
    
#     @classmethod
#     def create(cls, mod: 'Module', *args, **kwargs) -> Self:
#         """Classmethod to create a ModSrc with args and kwargs

#         Args:
#             mod (Module): The module

#         Returns:
#             Self: The ModSrc
#         """
#         return cls(
#             mod, NodeArgs(*args, **kwargs)
#         )


# def wait(
#     t: T, 
#     agg: typing.Callable[[typing.Any], typing.Any]=None
# ) -> T:
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
    
#     return T(val, WaitSrc(t, agg))


# class WaitSrc(Src):
#     """Indicates to wait until completed
#     """
#     def __init__(
#         self, incoming: T, 
#         agg: typing.Callable[[typing.Any], typing.Any]=None
#     ):
#         """Create a Src to wait for the incoming transmission

#         Args:
#             incoming (T): The incoming transmission
#         """
#         super().__init__()
#         self._incoming = incoming
#         self._agg = agg or (lambda x: x)

#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: The incoming Transmission
#         """
#         yield self._incoming

#     def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
#         """
#         Args:
#             by (typing.Dict[T, typing.Any]): The input to the network

#         Returns:
#             typing.Any: The output of the Src
#         """
#         by = by or {}
        
#         val = self._incoming.probe(by)

#         if isinstance(val, Partial) and not val.complete:
#             return WAITING
#         if isinstance(val, Partial):
#             return self._agg(val.full)

#         if isinstance(val, Streamer):
        
#             if not val.complete:
#                 res = val()
#                 if res.complete:
#                     return self._agg(res.full)
#                 return WAITING

#             return self._agg(val.output.full)
            
#         return val


# class Var(Src):
#     """A Variable Src that stores a default value.
#     """
    
#     def __init__(self, default=UNDEFINED, default_factory=UNDEFINED):
#         """A Variable Src

#         Args:
#             default (optional): The default value for the Src. Defaults to None.
#             default_factory (optional): A factory to retrieve the default. Defaults to None.
#         """
#         self.default = default
#         self.default_factory = default_factory
#         if default is UNDEFINED and default_factory is UNDEFINED:
#             raise RuntimeError('Either the default value or default factory must be defined')

#     def incoming(self) -> typing.Iterator[T]:
#         """
#         Yields:
#             Iterator[typing.Iterator[T]]: The incoming Transmissions - Since this is a variable Src it returns None
#         """
#         # hack to ensure it is a generator
#         if False:
#             yield False
        
#     def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
#         """Return the value of the variable

#         Args:
#             by (typing.Dict[T, typing.Any]): The input to the network

#         Raises:
#             RuntimeError: If the default value is not defined

#         Returns:
#             typing.Any: The value of the variable. If the 
#         """
#         by = by or {}
#         if self.default is not UNDEFINED:
#             return self.default
#         if self.default_factory is not UNDEFINED:
#             return self.default_factory()
#         raise RuntimeError('There is no default for var.')


# class IdxSrc(Src):
#     """Index the output of a transmission"""

#     def __init__(self, t: T, idx):
#         """Index the output of a transmission

#         Args:
#             t (T): The Transmission to index
#             idx: The index for the output of the transmission
#         """
#         self.t = t
#         self.idx = idx

#     def incoming(self) -> typing.Iterator['T']:
#         """
#         Yields:
#             T: The indexed transmission
#         """
#         yield self.t

#     def forward(self, by: typing.Dict[T, typing.Any]=None) -> typing.Any:
#         """
#         Args:
#             by (typing.Dict[T, typing.Any]): The input to the network

#         Returns:
#             typing.Any: The indexed value
#         """
#         by = by or {}
#         val = self.t.probe(by)
#         if is_undefined(val):
#             return val
#         return val[self.idx]


# def link(module: Module, *args, **kwargs) -> T:
#     """Create a link to a module

#     Args:
#         module (Module): The module to link to
#         *args: The arguments to the module
#         **kwargs: The keyword arguments to the module
#     """
#     args = NodeArgs(*args, **kwargs)
#     if not args.is_undefined():
#         partial = args.has_partial()
#         args = args.eval()
#         res = module.forward(*args.args, **args.kwargs)
#         if partial:
#             res = Partial(res)
#         return T(
#             res,
#             ModSrc(module, args)
#         )

#     return T(
#         UNDEFINED, ModSrc(module, args)
#     )


# # TODO: Combine the following two
# # TDOO: add async_link?
# # streamable
# def stream_link(module: 'StreamModule', *args, **kwargs) -> T:
#     """
#     Returns:
#         T: The Streamable transmission
#     """
#     args = NodeArgs(*args, **kwargs)
    
#     if not args.is_undefined():
#         args = args.eval()

#         # cur = module.stream(*args.args, **args.kwargs)
#         return T(
#             Streamer(module.stream(*args.args, **args.kwargs)),
#             StreamSrc(module, args)
#         )

#     return T(
#         UNDEFINED, StreamSrc(module, args)
#     )




# class Src(Module):
#     """Base class for Src. Use to specify how the
#     Transmission (T) was generated
#     """

#     @abstractmethod
#     def incoming(self) -> typing.Iterator['T']:
#         pass

#     @abstractmethod
#     def forward(
#         self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
#         pass

#     # def forward(self, by: typing.Dict['T', typing.Any]=None) -> typing.Any:
#     #     return self.forward(by)
