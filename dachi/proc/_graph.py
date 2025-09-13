
# 1st party
import typing
from typing import Self
import asyncio
from dataclasses import dataclass

# local
from ..utils import (
    is_undefined, UNDEFINED
)
from ._process import Process, AsyncProcess
from ..core import SerialDict
from dachi.core import ModuleDict, Attr
from dachi.core import AdaptModule
from dataclasses import dataclass
import typing as t

# TODO: Check if the value coming from incoming is undefined or waiting... 


class BaseNode(AsyncProcess):
    """Base class for nodes in the graph. It can be a Var or T.
    It is used to represent a node in the graph that can be processed or async processed.
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
    """A variable in the graph. A variable is a Root Node
    that feeds into T nodes (process nodes).
    
    It can be used to store a value that can be processed or async processed.
    It is used to represent a node in the graph that can be processed or async processed.
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
        by = by if by is not None else {}
        if (
            self in by 
            # and not isinstance(by[self], Streamer) 
            # and not isinstance(by[self], Partial)
        ):
            res = by[self]
        
        elif not is_undefined(self.val):
            res = self.val
        
        else: raise RuntimeError('Val hasnot been defined for Var')

        by[self] = res
        return res
    
    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        """Get the incoming connections for this node
        Since it is a Root node it has no incoming connections

        Yields:
            typing.Tuple[str, 'BaseNode']: The incoming connections
        """
        if False:
            yield None
        return


class T(BaseNode):
    """A process node in a DAG. It can be used to store a value that can be processed or async processed.
    It is both used to wrap a Process/AsyncProcess and to represent a node in the graph that can be processed or async processed.

    example:
        ```python
        t = T(
            args=SerialDict(),
            src=Process(),
            is_async=False
        )
        ```
        val = await t.aforward()

    Args:
        args (SerialDict): The arguments to the process
        src (Process | AsyncProcess): The process to execute
        is_async (bool, optional): Whether the process is async. Defaults to False.
    """
    args: SerialDict
    src: Process | AsyncProcess
    is_async: bool = False

    async def aforward(
        self, 
        by: dict['BaseNode', typing.Any] | None = None
    ) -> typing.Any:
        """Evaluate this node (memoised)."""
        by = by if by is not None else {}

        if self in by:
            return by[self]

        if not self.is_undefined():
            return self.val                       

        kw_serial = await self.get_incoming(by)   # SerialDict
        kwargs = dict(kw_serial.items())   

        if self.is_async:
            val = by[self] = await self.src(**kwargs)
        else:
            val = by[self] = self.src(**kwargs)
            print('Setting by[self]', by[self], self)

        if val is UNDEFINED:
            raise RuntimeError("Source returned UNDEFINED")

        return val

    def incoming(self) -> typing.Iterator[typing.Tuple[str, 'BaseNode']]:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self.args.items():
            if isinstance(arg, BaseNode):
                yield k, arg

    def eval_args(self) -> SerialDict:
        """Evaluate the current arguments
            - The value of t
            - The current value for a Streamer or Partial

        Returns:
            Self: Evaluate the input args to the process
        """
        kwargs = {}
        
        for k, a in self.args.items():
            if isinstance(a, BaseNode):
                a = a.val
            # elif isinstance(a, Partial):
            #     a = a.dx
            kwargs[k] = a
        return SerialDict(data=kwargs)

    # Have to evaluate the kwargs    
    async def get_incoming(
        self, 
        by: typing.Dict['T', typing.Any]=None
    ) -> typing.Dict[str, typing.Any]:
        """Get the incoming arguments for this node
        Args:
            by (typing.Dict[&#39;T&#39;, typing.Any], optional): The
                values to use for the incoming arguments.
                Defaults to None.
        Returns:
            typing.Dict[str, typing.Any]: The evaluated arguments
        """

        by = by if by is not None else {}
        kwargs = {}
        tasks = {}
        async with asyncio.TaskGroup() as tg:

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
        
        return SerialDict(data=kwargs)

    def has_partial(self) -> bool:

        for k, a in self.args.items():
            if isinstance(a, T):
                a = a.val
            # if isinstance(a, Partial) and not a.complete:
            #     return True
        return False

    def incoming(self) -> typing.Iterator['T']:
        """
        Yields:
            T: All of the arguments connected to another
            transmission
        """
        for k, arg in self.args.items():
            if isinstance(arg, BaseNode):
                yield arg


def t(
    p: Process, 
    _name: str=None, _annotation: str=None,
    **kwargs, 
) -> T:
    """Convenience function to create a T node from a Process. """

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


class CircularReferenceError(Exception):
    """Raised when a circular reference is detected in the DAG"""
    pass


@dataclass
class RefT:
    """Reference to a node in the DAG by name"""
    name: str


class DAG(AdaptModule, AsyncProcess):
    """Directed Acyclic Graph (DAG) for processing data.
    This class allows for the creation of a DAG where nodes can be processes or async processes.
    It supports the addition of nodes, setting of outputs, and forwarding of data through the graph

    RefT is used to reference nodes in the graph by name
    """

    def __post_init__(self):
        """Initialize the DAG with an empty set of nodes and outputs
        
        The args specify the args that get input into a node.
        They can be a reference (RefT) or a value. 
        If it is a reference, the node that is referenced will be resolved
        when the node is processed.

        Methods used in the DAG that are referenced by strings must be async.
        """
        super().__post_init__()
        # can be a "var"
        self._nodes = ModuleDict()
        self._args = Attr[typing.Dict[str, typing.Dict[str, RefT | typing.Any]]](data={})
        self._outputs = Attr[typing.List[str] | str](data=None)

    async def _sub(self, name: str, by: typing.Dict, visited: typing.Set[str]=None):
        """Subroutine to get the value of a node by name, resolving any references
        Args:
            name (str): The name of the node to resolve
            by (typing.Dict): A dictionary to store resolved nodes
        Returns:
            typing.Any: The resolved value of the node
        """
        if visited is None:
            visited = set()
        
        if name in visited:
            raise CircularReferenceError(f"Circular reference detected for node {name}")
        visited.add(name)
        node = self._nodes[name]
        args = self._args.data[name]

        if name in by:
            cur = by[name]
            return cur

        kwargs = {}

        async with asyncio.TaskGroup() as tg:
            for key, arg in args.items():
                if isinstance(arg, RefT):
                    # check
                    kwargs[key] = tg.create_task(
                        self._sub(arg.name, by, visited)
                    )
                else:
                    kwargs[key] = arg

        kwargs = {
            k: v if not isinstance(v, asyncio.Task) else v.result()
            for k, v in kwargs.items()
        }
        if isinstance(node, Process):
            res = node(**kwargs)
        elif isinstance(node, str):
            node = getattr(self, node, None)
            if node is None:
                raise ValueError(f"Node {node} not found in DAG")
            
            res = await node(**kwargs)
        elif isinstance(node, AsyncProcess):
            res = await node.aforward(**kwargs)
        else: 
            raise ValueError(f"Node {node} is not a Process or AsyncProcess")
        by[name] = res
        return res

    async def aforward(self, by: typing.Dict=None):
        """Forward the data through the graph, resolving all nodes and their arguments
        Args:
            by (typing.Dict, optional): A dictionary to store resolved nodes. Defaults to None.
        Returns:
            tuple: A tuple of resolved outputs from the graph
        """
        if self._outputs.data is None:
            return None
        by = by if by is not None else {}
        res = []
        for name in self._outputs.data:
            if name in by:
                res.append(by[name])
            else:
                res.append(await self._sub(name, by))

        if isinstance(self._outputs.data, str):
            return res[0]
        return tuple(res)


# TODO: Set up DAG deserialization to set
# the object if it is an FProc
class FProc(Process):
    """A process that executes a function and returns a status
    """

    name: str

    def __post_init__(self):
        """Initialize the FProc"""
        super().__post_init__()
        self.obj = None

    async def aforward(self, obj, kwargs) -> typing.Any:
        """Execute the process

        Returns:
            typing.Any: The result of the function execution
        """
        if self.status.is_done:
            return self.status
        
        if self.obj is None:
            raise ValueError(
                "Process object is not set. "
                "Please set the object before calling aforward."
            )

        f = getattr(self.obj, self.name, None)
        if f is None:
            raise ValueError(
                f"Function {self.name} not found in object {self.obj}"
            )
        return await f(**kwargs)
