# 1st party
import typing

# local
from ._core import (
    TakoBase, Var, T, to_by
)
from ._network import Network


class Tako(TakoBase):
    """Define a Graph Node that wraps multiple other nodes
    """

    def __init__(
        self, name: str, inputs: typing.List[Var], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._network = Network(outputs)
    
    def traverse(self, *args, **kwargs) -> typing.Iterator[T]:
        """Traverse each node in the Tako

        Returns:
            typing.Iterator[T]: An iterator which iterates over all the nodes
        """
        by = to_by(self._inputs, args, kwargs)
        for result in self._network.traverse(by):
            yield result
        
        # TODO: USE NETWORK
        # for result in traverse_ts(self._outputs, by, evaluate=False):
        #     yield result

    def op(self, *args, **kwargs) -> typing.Any:
        """
        """
        by = to_by(self._inputs, args, kwargs)
        return self._network.exec(by)
        # return probe_ts(self._outputs, by)
        # TODO: Use network
