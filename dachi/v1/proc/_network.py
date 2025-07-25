# 1st party
import typing

# 3rd party
import networkx as nx
from ._process import Module
from ._graph import T


class Network(object):
    """Object to wrap a network to use for forward or 
    reverse traversal
    """

    def _build_network(self, ts: typing.List[T], G: nx.DiGraph, all_ts: typing.Dict, incoming: typing.Dict):
        """Build a network of the nodes
        Args:
            ts (typing.List[T]): List of nodes to build the network from
            G (nx.DiGraph): Graph to build the network on
            all_ts (typing.Dict): Dictionary of all the nodes
            incoming (typing.Dict): Dictionary of the incoming nodes
        """
        for t in ts:
            t_id = id(t)
            if not G.has_node(t_id):
                G.add_node(t_id)
            inc_ts = []
            all_ts[t_id] = t
            incoming[t] = []
            for inc_i in t.src.incoming():
                
                if not G.has_node(id(inc_i.val)):
                    G.add_node(id(inc_i.val))

                G.add_edge(id(inc_i.val), t_id)
                inc_ts.append(inc_i.val)
                
                incoming[t_id].append(inc_i)
            self._build_network(inc_ts, G, all_ts, incoming)
                
    def __init__(self, outputs: typing.List[T]):
        """Instantiate a network
        Args:
            outputs (typing.List[T]): List of nodes to build the network from
        """
        super().__init__()
        # get all of the ts

        self._ts = {}
        self._incoming = {}
        self._G = nx.DiGraph()
        
        self._build_network(outputs, self._G, self._ts, self._incoming)
        self._execution_order = list(nx.topological_sort(self._G))
        self._outputs = outputs

    @property
    def G(self) -> nx.DiGraph:
        """Return the graph of the network
        Returns:
            nx.DiGraph: The graph of the network
        """
        return self._g

    def __len__(self) -> int:
        """Return the number of nodes in the network
        Returns:
            int: The number of nodes in the network
        """
        return len(self._execution_order)

    def __getitem__(self, idx: int) -> typing.Dict[T, typing.List[T]]:
        """Return the node at the given index
        Args:
            idx (int): The index of the node to return
        Returns:
            typing.Dict[T, typing.List[T]]: The node at the given index
        """
        name = self._execution_order[idx]
        return self._ts[name]
    
    def __iter__(self) -> typing.Iterator[T]:
        """Return an iterator over the nodes in the network
        Returns:
            typing.Iterator[T]: An iterator over the nodes in the network
        """
        for name in self._execution_order:
            yield self._ts[name]

    def incoming(self, t: T) -> typing.List[T]:
        """Return the incoming nodes for the given node
        Args:
            t (T): The node to get the incoming nodes for
        Returns:
            typing.List[T]: The incoming nodes for the given node
        """
        return self._incoming[id(t)]


class GraphAdapter(Module):
    """Define a Graph Node that wraps multiple other nodes
    """

    def __init__(
        self, inputs: typing.List[typing.Tuple[str, T]], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        """Instantiate a node which adapts the inputs to the outputs
        This is useful for wrapping multiple nodes into a single node

        Args:
            name (str): Name of the Adapter
            inputs (typing.List[Var]): Inputs 
            outputs (typing.List[typing.Union[typing.Tuple[T, int], T]]): 
        """
        super().__init__()
        self._inputs = inputs
        self._outputs = outputs

    def forward(self, *args, **kwargs) -> typing.Any:
        """Forward the inputs to the outputs
        Args:
            *args: The inputs to the node
            **kwargs: The inputs to the node
        Returns:
            typing.Any: The outputs of the node
        """
        by = {}
        defined = {}
        for arg, t in zip(args, self._inputs):
            by[t[1]] = arg
            defined[t[0]] = arg

        for k, v in kwargs.items():
            if k in defined:
                raise RuntimeError(f'Arg {k} has already been specified in the args')
            by[k] = arg 
        results = []
        for t in self._outputs:
            if isinstance(t, typing.Tuple):
                t, ind = t
            else:
                ind = None
            result = t.probe(
                by
            )
            if ind is not None:
                results.append(result[ind])
            else:
                results.append(result)

        return tuple(results)
