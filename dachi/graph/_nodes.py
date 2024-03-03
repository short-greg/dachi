# 1st party
import typing
from typing import Callable
import functools

# local
from ._core import Var, FieldList, Field, T, Node, to_by, Out


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
        fields = []
        self._out = Out(outputs)
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
        
        stored = to_by(self._inputs, args, kwargs)
        by = {}
        for t in self._out.ts:
            t(by, stored, deep=True)

        return self._out(stored)


def nodefunc(inputs: typing.List[Field], outputs: typing.List[Field]):
    """Decorator that wraps a function. Using this will convert the function to a NodeFunc

    Args:
        inputs (typing.List[Field]): The inputs to the function
        outputs (typing.List[Field]): The outputs from the function
    """
    def _(f):

        class NodeFunc(Node):
            """A Node that wraps a function
            """

            def __init__(self, f, inputs: typing.List[Field], outputs: typing.List[Field], instance):
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
                self._instance = instance
                self.op = functools.update_wrapper(self.op)

            @functools.wraps(f)
            def op(self, *args, **kwargs) -> typing.Any:
                
                if self._instance is not None:
                    return self.f(
                        self._instance, *args, **kwargs
                    )

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

        return NodeFunc(
            f, inputs, outputs
        )
    return _


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

    @property
    def inputs(self) -> FieldList:
        """
        Returns:
            FieldList: The outputs from the Node
        """

        return self._inputs


def nodedef(input_names, output_names):

    def _(f):
        f.__input_names__ = input_names
        f.__output_names__ = output_names
        return f
    return _


def linkf(f, *args, **kwargs):

    node = NodeFunc(
        f, f.__input_names__, f.__output_names__)
    return node.link(*args, **kwargs)

# class NodeMethod(Node):

#     def __init__(self, f: Callable, input_names, output_names, instance):

#         super().__init__(f.__name__)
#         self.f = f
#         self.input_names = input_names
#         self.output_names = output_names
#         self.instance = instance

#     def op(self, *args, **kwargs) -> typing.Any:
#         return self.f(self.instance, *args, **kwargs)
    

# add Async version
# class NodeMethodWrapper(object):

#     def __init__(self, f: Callable, input_names, output_names):

#         self.f = f
#         self.input_names = input_names
#         self.output_names = output_names

#     def op(self, *args, **kwargs) -> typing.Any:
#         return self.f(*args, **kwargs)
    
#     def __get__(self, instance, owner):
#         return NodeFunc(
#             self.f, self.input_names, self.output_names, instance
#         )
    

# def nodemethod(input_names, output_names):

#     def _(f):

#         return NodeMethodWrapper(f, input_names, output_names)
        
#     return _
