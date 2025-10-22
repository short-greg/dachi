"""
This module implements a Directed Acyclic Graph (DAG) for processing data using nodes that can be either variables (Var) or processes (T). There are two ways to implement this. 
1. Using the DAG class to define nodes and their connections explicitly.
2. Using the T and Var classes directly to create a graph structure.
Example usage:

var = Var(val=5, name='input')
t1 = T(
    val=UNDEFINED,
    src=SomeProcess(),
    args=SerialDict(items={'x': var})
)
t2 = T(
    val=UNDEFINED,
    src=AnotherProcess(),
    args=SerialDict(items={'y': t1})
)
# or t2 = t(AnotherProcess(), x=t1)
t2.probe(by={var: 10})  # Should return the output of AnotherProcess with input from SomeProcess with x=10


dag = DAG(
    nodes=ModuleDict(data={
        'input': var,
        't1': t1,
        't2': t2
    }),
    args=Attr(data={
        't1': {'x': RefT(name='input')},
        't2': {'y': RefT(name='t1')}
    }),
    outputs=Attr(data=['t2'])
)

result = dag.aforward(by={var: 10})  # Should return the output of AnotherProcess with input from SomeProcess with x=10
"""


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


@dataclass
class RefT:
    """Reference to a node in the DAG by name"""
    name: str


class DataFlow(AdaptModule, AsyncProcess):
    """DataFlow: Directed Acyclic Graph (DAG) for processing data pipelines.

    DataFlow is a declarative container for defining data processing pipelines using named
    nodes and references (RefT). The architecture prevents cycles by design - nodes can only
    reference nodes that already exist in the graph.

    Key Features:
        - Named nodes with RefT references for serialization
        - Parallel execution of independent nodes via asyncio.TaskGroup
        - Memoization - nodes evaluated once per execution
        - Support for both sync (Process) and async (AsyncProcess) nodes
        - String nodes for dynamic method dispatch
        - Output override for flexible execution
        - Sub-graph creation and node replacement

    Node Types:
        - Process/AsyncProcess: Computation nodes that execute logic
        - Var: Input/variable nodes (root nodes)
        - String: Method names on the DataFlow instance (called dynamically)

    Example:
        >>> from dachi.proc import DataFlow, RefT
        >>>
        >>> dag = DataFlow()
        >>> inp_ref = dag.add_inp('input', val=5)
        >>> square_ref = dag.link('square', Square(), x=inp_ref)
        >>> result_ref = dag.link('result', Add(), a=square_ref, b=inp_ref)
        >>> dag.set_out('result')
        >>>
        >>> output = await dag.aforward()  # Returns 30 (5^2 + 5)

    Note:
        This is a DAG (Directed Acyclic Graph) implementation. Circular references
        are prevented by the architecture and will cause runtime errors if forced.
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
        self._node_counter = Attr[int](data=0)
        self._var_counter = Attr[int](data=0)

    def _generate_node_name(self, prefix: str = "node") -> str:
        """Generate unique node name with given prefix

        Args:
            prefix (str): The prefix for the generated name
        Returns:
            str: A unique node name
        """
        if prefix == "node":
            name = f"{prefix}_{self._node_counter.data}"
            self._node_counter.data += 1
        elif prefix == "var":
            name = f"{prefix}_{self._var_counter.data}"
            self._var_counter.data += 1
        else:
            counter = 0
            while f"{prefix}_{counter}" in self._nodes:
                counter += 1
            name = f"{prefix}_{counter}"
        return name

    async def _sub(self, name: str, by: typing.Dict, visited: typing.Dict[str, asyncio.Task] | None = None):
        """Subroutine to get the value of a node by name, resolving any references
        Args:
            name (str): The name of the node to resolve
            by (typing.Dict): A dictionary to store resolved nodes
        Returns:
            typing.Any: The resolved value of the node
        """
        if visited is None:
            visited = dict()

        if name in by:
            return by[name]

        if name in visited:
            task = visited[name]
            current_task = asyncio.current_task()
            if task is not current_task:
                if not task.done():
                    await task
                return task.result()
            elif name in by:
                return by[name]

        node = self._nodes[name]
        args = self._args.data[name]

        kwargs = {}

        async with asyncio.TaskGroup() as tg:
            for key, arg in args.items():
                if isinstance(arg, RefT):
                    task = tg.create_task(
                        self._sub(arg.name, by, visited)
                    )
                    if arg.name not in visited:
                        visited[arg.name] = task
                    kwargs[key] = task
                else:
                    kwargs[key] = arg

        kwargs = {
            k: v if not isinstance(v, asyncio.Task) else v.result()
            for k, v in kwargs.items()
        }
        if isinstance(node, Process):
            res = node(**kwargs)
        elif isinstance(node, AsyncProcess):
            res = await node.aforward(**kwargs)
        elif isinstance(node, str):
            method = getattr(self, node, None)
            if method is None:
                raise ValueError(
                    f"Method {node} not found in {type(self).__name__}"
                )
            res = await method(**kwargs)
        else:
            raise ValueError(
                f"Node {name} is not a Process or AsyncProcess"
            )
        
        by[name] = res
        return res
    
    def link(self, name: str, node: Process | AsyncProcess, **kwargs: RefT | typing.Any) -> RefT:
        """Link a computation node to the DataFlow.

        Adds a Process or AsyncProcess node to the graph with the given name. Arguments
        can be either literal values or RefT references to other nodes. When the DataFlow
        is executed, dependencies will be resolved automatically.

        Args:
            name (str): Unique name for this node. Must not already exist in the DataFlow.
            node (Process | AsyncProcess): The computation node to execute.
            **kwargs: Arguments to pass to the node. Can be literal values or RefT references
                     to other nodes in the graph.

        Returns:
            RefT: A reference to this node that can be used in other node arguments.

        Raises:
            ValueError: If a node with this name already exists.

        Example:
            >>> dag = DataFlow()
            >>> inp = dag.add_inp('x', val=10)
            >>> doubled = dag.link('double', Multiply(), value=inp, factor=2)
            >>> dag.link('result', Add(), a=doubled, b=5)
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in DAG")
        self._nodes[name] = node
        self._args.data[name] = kwargs
        return RefT(name=name)
    
    def add_inp(self, name: str, val: typing.Any) -> RefT:
        """Add an input variable (root node) to the DataFlow.

        Input variables are root nodes that provide initial values to the graph.
        They can be overridden at execution time via the `by` parameter in aforward().

        Args:
            name (str): Unique name for this input. Must not already exist in the DataFlow.
            val (typing.Any): The default value for this input.

        Returns:
            RefT: A reference to this input that can be used in other node arguments.

        Raises:
            ValueError: If a node with this name already exists.

        Example:
            >>> dag = DataFlow()
            >>> x_ref = dag.add_inp('x', val=5)
            >>> y_ref = dag.add_inp('y', val=10)
            >>> result = dag.link('sum', Add(), a=x_ref, b=y_ref)
            >>> dag.set_out('sum')
            >>> await dag.aforward()  # Uses default values: 5 + 10 = 15
            >>> await dag.aforward(by={dag._nodes['x']: 20})  # Override x: 20 + 10 = 30
        """
        if name in self._nodes:
            raise ValueError(f"Node {name} already exists in DAG")
        self._nodes[name] = Var(val=val, name=name)
        self._args.data[name] = {}
        return RefT(name=name)

    def set_out(self, outputs: typing.List[str]|str) -> None:
        """Set the output nodes of the DataFlow.

        Specifies which nodes should be returned when executing the DataFlow.
        If a single string is provided, aforward() returns that node's value directly.
        If a list is provided, aforward() returns a tuple of values.

        Args:
            outputs (typing.List[str]|str): Names of nodes to output. Can be a single
                                           node name or a list of node names.

        Raises:
            ValueError: If any specified node doesn't exist in the DataFlow.

        Example:
            >>> dag.set_out('result')  # Single output
            >>> await dag.aforward()  # Returns single value
            >>>
            >>> dag.set_out(['x', 'y', 'z'])  # Multiple outputs
            >>> await dag.aforward()  # Returns tuple (x_val, y_val, z_val)
        """
        output_list = outputs if isinstance(outputs, list) else [outputs]
        for output in output_list:
            if output not in self._nodes:
                raise ValueError(f"Output node '{output}' does not exist in DataFlow")
        self._outputs.data = outputs

    def __contains__(self, item: str) -> bool:
        """Check if the DAG contains a node with the given name
        Args:
            item (str): The name of the node to check
        Returns:
            bool: True if the node exists, False otherwise
        """
        return item in self._nodes
    
    def sub(self, outputs: typing.List[str], by: typing.Dict[str, typing.Any]) -> 'DataFlow':
        """Create a sub-DAG with the given outputs
        Args:
            outputs (typing.List[str]): The names of the output nodes
            by (typing.Dict[str, typing.Any]): A dictionary to store resolved nodes
        Returns:
            DAG: The sub-DAG
        """
        sub_dag = DataFlow()
        for name in outputs:
            if name not in self._nodes:
                raise ValueError(f"Node {name} does not exist in DAG")
            sub_dag._nodes[name] = self._nodes[name]
            sub_dag._args.data[name] = self._args.data[name]
        sub_dag.set_out(outputs)
        return sub_dag
        
    def replace(self, name: str, node: Process | AsyncProcess) -> None:
        """Replace a node in the DAG
        Args:
            name (str): The name of the node to replace
            node (Process | AsyncProcess): The new node
        """
        if name not in self._nodes:
            raise ValueError(f"Node {name} does not exist in DAG")
        self._nodes[name] = node

    async def aforward(
        self,
        by: typing.Dict=None,
        out_override: typing.List[str]|str|RefT=None
    ):
        """Execute the DataFlow and return the output values.

        Resolves all dependencies and executes nodes in the correct order. Independent
        nodes are executed in parallel using asyncio.TaskGroup. Each node is evaluated
        only once per execution (memoization via the `by` dict).

        Args:
            by (typing.Dict, optional): Dictionary mapping Var nodes to override values.
                                       Use to provide different inputs at execution time.
            out_override (typing.List[str]|str|RefT, optional): Override the default outputs
                                                                for this execution only.

        Returns:
            Single value if outputs is a string, tuple of values if outputs is a list,
            None if no outputs are set.

        Example:
            >>> dag = DataFlow()
            >>> x = dag.add_inp('x', val=5)
            >>> result = dag.link('square', Square(), x=x)
            >>> dag.set_out('square')
            >>>
            >>> await dag.aforward()  # Returns 25
            >>> await dag.aforward(by={dag._nodes['x']: 10})  # Returns 100
            >>> await dag.aforward(out_override='x')  # Returns 5 (outputs x instead)
        """
        outputs = out_override if out_override is not None else self._outputs.data

        if outputs is None:
            return None

        if isinstance(outputs, (str, RefT)):
            outputs = [outputs]
            single = True
        else:
            single = False

        by = by if by is not None else {}
        res = []

        for output in outputs:
            if isinstance(output, RefT):
                name = output.name
            else:
                name = output

            if name in by:
                res.append(by[name])
            else:
                res.append(await self._sub(name, by))

        if single:
            return res[0]
        return tuple(res)
    
    @classmethod
    def from_node_graph(cls, nodes: typing.List[BaseNode]):
        """Create a DAG from a list of nodes

        Args:
            nodes (typing.List[BaseNode]): The nodes to create the DAG from 
        Returns:
            DAG: The created DAG
        """
        dag = cls()
        for node in nodes:
            if node.name is None:
                raise ValueError("Node must have a name to be added to DAG")
            if isinstance(node, Var):
                dag.add_inp(name=node.name, val=node.val)
            elif isinstance(node, T):
                args = {}
                for k, arg in node.args.items():
                    if isinstance(arg, BaseNode):
                        args[k] = RefT(name=arg.name)
                    else:
                        args[k] = arg
                dag.link(name=node.name, node=node.src, **args)
            else:
                raise ValueError("Node must be a Var or T to be added to DAG")
        return dag
    
    def to_node_graph(self) -> typing.List[BaseNode]:
        """Convert the DAG to a list of nodes

        Returns:
            typing.List[BaseNode]: The list of nodes
        """
        nodes = []
        for name, node in self._nodes.items():
            if isinstance(node, Var):
                nodes.append(Var(val=node.val, name=name))
            elif isinstance(node, (Process, AsyncProcess)):
                args = {}
                for k, arg in self._args.data[name].items():
                    if isinstance(arg, RefT):
                        ref_node = self._nodes[arg.name]
                        args[k] = ref_node
                    else:
                        args[k] = arg
                is_async = isinstance(node, AsyncProcess)
                nodes.append(
                    T(
                        src=node,
                        args=SerialDict(data=args),
                        name=name,
                        is_async=is_async
                    )
                )
            else:
                raise ValueError("Node must be a Var or T to be converted from DAG")
        return nodes


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
