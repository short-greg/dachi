import typing

from .core import Node, Var, FieldList, Field, probe_ts, to_by, T


class Adapter(Node):

    def __init__(
        self, name: str, inputs: typing.List[Var], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        """Instantiate a node which adaptas

        Args:
            name (str): Name of the Adapter
            inputs (typing.List[Var]): Inputs 
            outputs (typing.List[typing.Union[typing.Tuple[T, int], T]]): _description_
        """
        super().__init__(name)
        self._inputs = inputs
        self._outputs = outputs
        fields = []
        for output in outputs:
            if isinstance(output, tuple):
                output = output[0]
            fields.append(Field(output.name, default=output.value))
        self._output_fields = FieldList(fields)
    
    @property
    def outputs(self) -> FieldList:
        """Retrieve the fields

        Returns:
            FieldList: 
        """
        return self._output_fields

    def op(self, *args, **kwargs) -> typing.Any:
        
        by = to_by(self._inputs, args, kwargs)
        return probe_ts(self._outputs, by)
