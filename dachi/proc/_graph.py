
# 1st party
from abc import abstractmethod, ABC
import typing
import time
from typing import Self
import asyncio

# local
from ..utils import (
    is_undefined, UNDEFINED, WAITING,
    is_async_function
)
from ._process import Process, AsyncProcess
from ..core import BaseModule, SerialDict
from ._process import Partial, StreamProcess, AsyncStreamProcess


# TODO: Check if the value coming from incoming is undefined or waiting... 

class BaseNode(AsyncProcess):
    """
    """

    name: str | None = None
    val: typing.Any = UNDEFINED
    annotation: str | None = None
    
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

    def is_undefined(self) -> bool:
        """
        Returns:
            bool: Whether the transmission is Undefineed (WAITING or UNDEFINED)
        """
        return is_undefined(self.val)

    def __getitem__(self, idx: int) -> 'BaseNode':
        """
        Args:
            idx (int): The index to use on the transmission

        Returns:
            T: A transmission indexed.
        """
        # TODO: Add multi-indices
        if is_undefined(self.val):
            return T(
                val=self.val, src=Idx(idx=idx),
                args=SerialDict()
            )

        return T(
            val=self.val[idx], src=Idx(idx=idx), 
            args=SerialDict()
        )
    
    def detach(self) -> typing.Self:
        """Remove the Src from T. This "detaches" T from the network

        Returns:
            T: The detached T
        """
        return T(
            val=self.val, src=None, args=SerialDict()
        )
    
    async def aforward(
        self, by: typing.Dict['T', typing.Any]=None
    ) -> typing.Any:
        pass

    def incoming(self) -> typing.Iterator['BaseNode']:
        if False:
            yield None


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

        if (
            self in by 
            # and not isinstance(by[self], Streamer) 
            and not isinstance(by[self], Partial)
        ):
            res = by[self]
        
        elif not is_undefined(self.val):
            res = self.val
        
        else: raise RuntimeError('Val hasnot been defined for Var')

        by[self] = res
        return res
    
    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        
        if False:
            yield None
        return


class ProcNode(BaseNode):
    """    A node that wraps a process and its arguments
    """

    args: SerialDict

    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self.args.data():
            if isinstance(arg, BaseNode):
                yield k, arg

    def has_partial(self) -> bool:

        for k, a in self.args.data():
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
        if self.is_undefined():
            return None
        kwargs = {}
        
        for k, a in self.args.data():
            if isinstance(a, BaseNode):
                a = a.val
            elif isinstance(a, Partial):
                a = a.dx
            kwargs[k] = a
        return SerialDict(data=kwargs)

    # Have to evaluate the kwargs    
    async def get_incoming(
        self, 
        by: typing.Dict['T', typing.Any]=None
    ) -> Self:

        by = by or {}
        kwargs = {}
        tasks = {}
        async with asyncio.TaskGroup() as tg:

            for k, arg in self.args.data():
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
        
        return SerialDict(data=kwargs)

    def has_partial(self) -> bool:

        for k, a in self.args.data():
            if isinstance(a, T):
                a = a.val
            if isinstance(a, Partial) and not a.complete:
                return True
        return False

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self.args.data():
            if isinstance(arg, BaseNode):
                yield arg


class T(ProcNode):
    """...
    """
    src: Process | AsyncProcess
    is_async: bool = False

    async def aforward(self, by: dict['BaseNode', typing.Any] | None = None) -> typing.Any:
        """Evaluate this node (memoised)."""
        by = by or {}

        # ❶ already in probe-dict?
        if self in by:
            return by[self]

        # ❷ cached from previous run?
        if not self.is_undefined():
            return self.val                       # ← uses .val

        # ❸ gather upstream arguments
        kw_serial = await self.get_incoming(by)   # SerialDict
        kwargs = dict(kw_serial.items())          # plain dict for call-site

        # ❹ call wrapped process (sync / async)
        if self.is_async:
            self.val = by[self] = await self.src(**kwargs)
        else:
            self.val = by[self] = self.src(**kwargs)

        # ❺ sanity-check
        if self.val is UNDEFINED:
            raise RuntimeError("Source returned UNDEFINED")

        return self.val

    # async def aforward(
    #     self, by: typing.Dict['BaseNode', typing.Any]=None
    # ) -> typing.Any:
    #     """Probe the graph using the values specified in by

    #     Args:
    #         by (typing.Dict[&#39;T&#39;, typing.Any]): The inputs to the network

    #     Raises:
    #         RuntimeError: If the value is undefined

    #     Returns:
    #         typing.Any: The value returned by the probe
    #     """
    #     by = by or {}
    #     if not is_undefined(self.val):
    #         return self.val

    #     if self in by:
    #         return by[self]
            
    #     args, kwargs = self.args(by)
        
    #     if self._is_async:
    #         val = by[self] = await self.src(*args, **kwargs)
    #     else:
    #         val = by[self] = self.src(*args, **kwargs)    

    #     if val is UNDEFINED:
    #         raise RuntimeError(
    #             'Val has not been defined and no source for T'
    #         )
    #     return val


class Stream(ProcNode):
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
        if not is_undefined(self.val):
            return self.val
        
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
            self.val = by[self] = Partial(
                dx=self._dx, complete=False, prev=self._prev,
                full=self._full
            )
        except StopIteration:
            self.val = by[self] = Partial(
                dx=None, complete=True,
                prev=self._dx, full=self._full
            )

        return self.val


def t(
    p: Process, 
    _name: str=None, _annotation: str=None,
    **kwargs, 
) -> T:

    args = SerialDict(data=kwargs)
    return T(
        src=p, args=args, name=_name, annotation=_annotation,
        is_async=False
    )


def async_t(
    p: AsyncProcess,
    _name: str=None, _annotation: str=None,
    **kwargs, 
) -> T:

    args = SerialDict(data=kwargs)
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
