import typing

from ._core import (
    Tako, Var, T, to_by, 
    traverse_ts, probe_ts
)


class TakoWrapper(Tako):
    """Define a Graph Node that wraps multiple other nodes
    """

    def __init__(
        self, name: str, inputs: typing.List[Var], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
    
    def traverse(self, *args, **kwargs) -> typing.Iterator[T]:
        """Traverse each node in the Tako

        Returns:
            typing.Iterator[T]: An iterator which iterates over all the nodes
        """
        by = to_by(self._inputs, args, kwargs)
        for result in traverse_ts(self._outputs, by, evaluate=False):
            yield result

    def op(self, *args, **kwargs) -> typing.Any:
        """
        """

        by = to_by(self._inputs, args, kwargs)
        return probe_ts(self._outputs, by)
