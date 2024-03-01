# 1st party
import typing
from typing import Callable
from abc import ABC, abstractmethod, abstractproperty

# local
from ._core import Var, FieldList, Field, T, Node, to_by


# TODO: Review decorators to improve this
# class NodeMethod(Node):
#     """A node that wraps a method
#     """

#     def __init__(self, obj, f, inputs: typing.List[Field], outputs: typing.List[Field]):
#         """Create a Node out of an instance method

#         Args:
#             obj: The object wrapped
#             f: The function to call
#             inputs (typing.List[Field]): The input fields
#             outputs (typing.List[Field]): The output fields
#         """
#         super().__init__(f'{obj.__class__.__name__}_{f.__name__}')
#         self._obj = obj
#         self._f = f
#         self._inputs = inputs
#         self._outputs = outputs

#     def op(self, *args, **kwargs) -> typing.Any:
#         """Call the method in the object

#         Returns:
#             typing.Any: The output of the function
#         """
#         return self._f(self._obj, *args, **kwargs)


class Adapter(Node):
    """A Node which wraps a graph
    """

    def __init__(
        self, name: str, inputs: typing.List[Var], 
        outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        """Instantiate a node which adaptas

        Args:
            name (str): Name of the Adapter
            inputs (typing.List[Var]): Inputs 
            outputs (typing.List[typing.Union[typing.Tuple[T, int], T]]): 
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

        result = []
        stored = {}
        for output in self._outputs:
            
            if isinstance(output, T):
                result.append(output(by, stored))
            else:
                cur_result = output[0](by, stored)
                result.append(cur_result[output[1]])
        return result

class NodeFunc(Node):
    """A Node that wraps a function
    """

    def __init__(self, f, inputs: typing.List[Field], outputs: typing.List[Field]):
        """Initialize a 

        Args:
            f: The function to wrap
            inputs (typing.List[Field]): The inputs to the function
            outputs (typing.List[Field]): The outputs fromthe function
        """
        
        super().__init__(name=f.__name__)
        self.f = f
        self._inputs = inputs
        self._outputs = outputs

    def op(self, *args, **kwargs) -> typing.Any:
        
        return self.f(
            *args, **kwargs
        )

    @property
    def outputs(self) -> FieldList:
        """
        Returns:
            FieldList: The outputs from the Node
        """

        return self._outputs


def nodefunc(inputs: typing.List[Field], outputs: typing.List[Field]):
    """Decorator that wraps a function. Using this will convert the function to a NodeFunc

    Args:
        inputs (typing.List[Field]): The inputs to the function
        outputs (typing.List[Field]): The outputs from the function
    """

    def _(f):

        return NodeFunc(
            f, inputs, outputs
        )
    return _


class NodeMethod(Node):

    def __init__(self, f: Callable, input_names, output_names, instance):

        super().__init__(f.__name__)
        self.f = f
        self.input_names = input_names
        self.output_names = output_names
        self.instance = instance

    def op(self, *args, **kwargs) -> typing.Any:
        return self.f(self.instance, *args, **kwargs)
    

# add Async version
class NodeMethodWrapper(object):

    def __init__(self, f: Callable, input_names, output_names):

        self.f = f
        self.input_names = input_names
        self.output_names = output_names

    def op(self, *args, **kwargs) -> typing.Any:
        return self.f(*args, **kwargs)
    
    def __get__(self, instance, owner):
        # def wrapper(*args, **kwargs):

        #     return self.f(instance, *args, **kwargs)
        return NodeMethod(
            self.f, self.input_names, self.output_names, instance
        )

def nodemethod(input_names, output_names):

    def _(f):

        return NodeMethodWrapper(f, input_names, output_names)
        
    return _
