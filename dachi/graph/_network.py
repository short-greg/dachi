# 1st party
import typing

# 3rd party
import networkx as nx
from .._core._process import Module
from ._core import T


class Network(object):
    """Object to wrap a network to use for forward or 
    reverse traversal
    """

    def _build_network(self, ts: typing.List[T], G: nx.DiGraph, all_ts: typing.Dict, incoming: typing.Dict):

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
        # get all of the ts

        self._ts = {}
        self._incoming = {}
        self._G = nx.DiGraph()
        
        self._build_network(outputs, self._G, self._ts, self._incoming)
        self._execution_order = list(nx.topological_sort(self._G))
        self._outputs = outputs

    @property
    def G(self) -> nx.DiGraph:
        return self._g

    def __len__(self) -> int:
        return len(self._execution_order)

    def __getitem__(self, idx: int) -> typing.Dict[T, typing.List[T]]:
        name = self._execution_order[idx]
        return self._ts[name]
    
    def __iter__(self) -> typing.Iterator[T]:

        for name in self._execution_order:
            yield self._ts[name]

    def incoming(self, t: T) -> typing.List[T]:
    
        return self._incoming[id(t)]


class TAdapter(Module):
    """A Node which wraps a graph
    """

    def __init__(
        self, inputs: typing.List[typing.Tuple[str, T]], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        """Instantiate a node which adaptas

        Args:
            name (str): Name of the Adapter
            inputs (typing.List[Var]): Inputs 
            outputs (typing.List[typing.Union[typing.Tuple[T, int], T]]): 
        """
        super().__init__()
        self._inputs = inputs
        self._outputs = outputs

    def forward(self, *args, **kwargs) -> typing.Any:
        
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



# class Tako(TakoBase):
#     """Define a Graph Node that wraps multiple other nodes
#     """

#     def __init__(
#         self, name: str, inputs: typing.List[TIn], 
#         outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
#     ):
#         self._name = name
#         self._inputs = inputs
#         self._outputs = outputs
#         self._network = Network(outputs)
    
#     def traverse(self, *args, **kwargs) -> typing.Iterator[T]:
#         """Traverse each node in the Tako

#         Returns:
#             typing.Iterator[T]: An iterator which iterates over all the nodes
#         """
#         by = to_by(self._inputs, args, kwargs)
#         for result in self._network.traverse(by):
#             yield result
        
#         # TODO: USE NETWORK
#         # for result in traverse_ts(self._outputs, by, evaluate=False):
#         #     yield result

#     def forward(self, *args, **kwargs) -> typing.Any:
#         """
#         """
#         by = to_by(self._inputs, args, kwargs)
#         return self._network.exec(by)
#         # return probe_ts(self._outputs, by)
#         # TODO: Use network

