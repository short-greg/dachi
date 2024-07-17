# 1st party
import typing

# 3rd party
import networkx as nx

from ._core2 import T


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
