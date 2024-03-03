# 1st party
import typing

# 3rd party
import networkx as nx
from ._core import Incoming, T, Var, Out


class Network(object):

    def _traverse(self, ts: typing.List[T], G: nx.DiGraph, all_ts: typing.Dict, incoming: typing.Dict):

        for t in ts:
            if not G.has_node(t.id):
                G.add_node(t.id)
            inc_ts = []
            all_ts[t.id] = t
            incoming[t.id] = []
            for inc_i in t.incoming:
                if inc_i.is_t():
                    if not G.has_node(inc_i.val.id):
                        G.add_node(inc_i.val.id)

                    G.add_edge(inc_i.val.id, t.id)
                    inc_ts.append(inc_i.val)
                
                incoming[t.id].append(inc_i)
            self._traverse(inc_ts, G, all_ts, incoming)
                
    def __init__(self, outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]):
        # get all of the ts

        self._ts = {}
        self._incoming = {}
        self._G = nx.DiGraph()
        
        self._out = Out(outputs)
        self._traverse(self._out.ts, self._G, self._ts, self._incoming)
        self._execution_order = list(nx.topological_sort(self._G))
        self._outputs = outputs

    def __len__(self) -> int:
        return len(self._execution_order)

    def __getitem__(self, idx: int) -> typing.Dict[T, typing.List[Incoming]]:
        name = self._execution_order[idx]
        return self._ts[name]
    
    def incoming(self, t: typing.Union[T, str]) -> typing.List[Incoming]:

        if isinstance(t, T):
            t = t.id
        return self._incoming[t]
    
    def __iter__(self) -> typing.Iterator[T]:

        for name in self._execution_order:
            yield self._ts[name]
    
    def exec(self, by: typing.Dict[Var, typing.Any]):

        stored = {}
        for t in self:
            print(t.id)
            t(by, stored, deep=False)

        return self._out.__call__(stored)
    
    # will need to add async, stream, async_stream etc