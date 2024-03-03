# 1st party
from abc import abstractmethod, ABC, abstractproperty
import typing
from dataclasses import dataclass
import asyncio

# 3rd party
import networkx as nx


def _is_function(f) -> bool:
    """ 
    Args:
        f: The value to check

    Returns:
        bool: whether f is a function
    """
    f_type = type(f)
    return f_type == type(_is_function) or f_type == type(hasattr)


class F(object):
    """F is a functor that allows the user to set the args and kwargs
    """

    def __init__(self, f: typing.Callable[[], typing.Any], *args, **kwargs):
        """Create a functor

        Args:
            f (typing.Callable[[], typing.Any]): The function called
        """
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    @property
    def value(self) -> typing.Any:
        return self.f(*self.args, **self.kwargs)


class Field(object):
    """Specify the fields for a node
    """

    def __init__(self, name: str, dtype: str=None, default: typing.Union[typing.Any, typing.Callable[[], typing.Any]]=None):
        """Create a field object

        Args:
            name (str): Name of the field
            dtype (str, optional): Type of the field. Defaults to None.
            default (typing.Union[typing.Any, typing.Callable[[], typing.Any]], optional): Default value for the field. Defaults to None.
        """
        self.name = name
        self.dtype = dtype
        self.default = default

    @classmethod
    def factory(self, f) -> 'Field':
        """Create a field based on a function

        Args:
            f: The function or str or Field to wrap

        Returns:
            Field: The instantiated field
        """
        if isinstance(f, Field):
            return Field(f.name, f.dtype, f.default)
        if isinstance(f, str):
            return Field(f)
        if isinstance(f, typing.Tuple) or isinstance(f, typing.List):
            return Field(*f)
        raise ValueError(f'Argument f must be of type Field, string or tuple')


class FieldList(object):
    """Define a list of fields
    """

    def __init__(self, fields: typing.List[Field]):
        """Create A FieldList with a list of fields.

        Args:
            fields (typing.List[Field]): Fields to create with
        """

        self.fields = [Field.factory(field) for field in fields]
        self._field_map = {field.name: i for i, field in enumerate(self.fields)}

    def index(self, name: str) -> int:
        """Retrieve the index of the field

        Args:
            name (str): name of the field

        Raises:
            ValueError: If the index does not exist

        Returns:
            int: Index of the field
        """
        try:
            return self._field_map[name]
        except KeyError:
            raise ValueError(f'No field named {name} in the Field List')


class T(ABC):
    """Base transmission class"""

    def __init__(self, name: str):
        """Create a transmission from one node to another

        Args:
            name (str): Name of the transmission
        """
        self._id = id(self)
        self._name = name

    @property
    def id(self) -> int:
        return self._id

    @property
    def value(self) -> typing.Any:
        """
        Returns:
            typing.Any: The output of the transmission
        """
        pass

    def __getitem__(self, idx) -> 'TIdx':
        return TIdx(self, idx)
    
    @property
    def is_output(self) -> bool:
        return False

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the transmission
        """
        return self._name

    def __call__(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, deep: bool=True) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        pass

    @abstractmethod
    def clone(self) -> 'T':
        pass

    @abstractproperty
    def incoming(self) -> typing.List['Incoming']:
        pass


class Var(T):
    """Create a variable transmission that will output a value to send to nodes
    """

    def __init__(self, name: str, dtype: str=None, default:typing.Any=None):
        """

        Args:
            name (str): Name of the variable
            dtype (str, optional): The type of the variable. Defaults to None.
            default (optional): Default variable for the variable. Defaults to None. If it is of type function or
             builtin_function_or_method it will be called
        """
        super().__init__(name)
        self.dtype = dtype
        self._default = default

    @property
    def value(self) -> typing.Any:
        """
        Returns:
            typing.Any: The default value for the var
        """
        if isinstance(self._default, F):
            return self._default.value
        return self._default

    def set(self, default):
        """

        Args:
            default: Set the default value for the var
        """
        self._default = default

    def validate(self, by: typing.Dict['Var', typing.Any]) -> typing.Any:
        """Validate the value in by

        Args:
            by (typing.Dict[Var, typing.Any]): List of variables and their values

        Raises:
            ValueError: if the value is not is not valid

        Returns:
            typing.Any
        """

        try:
            val = by[self]
        except KeyError:
            return self.value
        if self.dtype is not None and not isinstance(val, self.dtype):
            raise ValueError(f'Value must be of dtype {self.dtype}')
        return val        

    def clone(self) -> 'Var':
        return Var(
            self.name, self.dtype, self._default
        )

    def __call__(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, deep: bool=True) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        return by.get(self, self._default)

    @property
    def incoming(self) -> typing.List['Incoming']:
        return []


@dataclass
class Incoming:

    val: typing.Union[T, typing.Any]
    idx: typing.Union[int, str]

    def is_t(self) -> bool:

        return isinstance(self.val, T)
    
    @classmethod
    def _process_arg(cls, t: T, by: typing.Dict[Var, typing.Any], 
        stored: typing.Dict[str, typing.Any]):
        
        if not isinstance(t, T):
            return t
        if t in by:
            return by[t]
        if t.id in stored:
            return stored[t.id]
        return t
        
    @classmethod
    def prepare_args(
        cls, incoming: typing.List['Incoming'], by: typing.Dict[Var, typing.Any], 
        stored: typing.Dict[str, typing.Any]
    ):
        args = []
        kwargs = {}
        for inc_i in incoming:
            arg = cls._process_arg(inc_i.val, by, stored)
            if isinstance(inc_i.idx, str):
                kwargs[inc_i.idx] = arg
            elif inc_i.idx >= len(args):
                args.append(arg)
            else:
                args.insert(inc_i.idx, arg)

        return args, kwargs


class OpStream:
    
    @abstractproperty
    def result(self) -> typing.Any:
        pass

    @abstractproperty
    def delta(self) -> typing.Any:
        pass


class SingleOpStream:

    def __init__(self, value):
        self.value = value

    def result(self) -> typing.Any:
        
        return self.value

    def delta(self) -> typing.Any:
        
        return self.value


class Node(ABC):
    """Defines a node in the graph. Use it to implement an operation
    """

    outputs = FieldList([])

    def __init__(self, name: str):
        """Create the node

        Args:
            name (str): Name of the node
        """
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the node
        """
        return self._name

    @abstractmethod
    def op(self) -> typing.Any:
        """The operation for the node. 

        Returns:
            typing.Any: The result of the node
        """
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:
        """Convenience method that wraps op()

        Returns:
            typing.Any: The output of op()
        """
        return self.op(*args, **kwargs)
    
    async def async_op(self, *args, **kwargs) -> typing.Any:
        
        return self.op(*args, **kwargs)
    
    def stream(self, *args, **kwargs) -> typing.Iterator[OpStream]:
        
        result = self.op(*args, **kwargs)
        yield SingleOpStream(result)

    async def async_stream(self, *args, **kwargs) -> typing.AsyncGenerator[OpStream, None]:
        
        for result in self.stream(*args, **kwargs):
            yield result

    def link(self, *args, **kwargs) -> typing.Union[T, typing.Tuple[T], typing.Any, typing.Tuple[typing.Any]]:
        """Use to link transmissions together. This wraps the op method. It can take either
        a transmission as an input or it cna take values

        Returns:
            typing.Union[typing.Any, typing.Tuple[typing.Any]]: 
        """
        is_variable = False

        op_args = []
        op_kwargs = {}
        for arg in args:
            arg, is_variable = get_arg(arg, is_variable)
            op_args.append(arg)

        for k, arg in op_kwargs.items():
            arg, is_variable = get_arg(arg, is_variable)
            op_kwargs[k] = arg

        if is_variable:
            return Process(self, args, op_kwargs)

        args = [arg.value if isinstance(arg, T) else arg for arg in args]
        kwargs = {k: arg.value if isinstance(arg, T) else arg for k, arg in kwargs.items()}
        result = self.op(*args, **kwargs)
        if isinstance(result, typing.Tuple):
            return tuple(Output(self._name, result_i, self) for result_i in result)
        return Output(self._name, result, self)
    
    def clone(self) -> 'Node':
        
        return self.__class__(
            self.name
        )


class Tako(Node):
    """Define a Graph Node that wraps multiple other nodes
    """
    
    @abstractmethod
    def traverse(self, *args, **kwargs) -> typing.Iterator[T]:
        """Traverse each node in the Tako

        Returns:
            typing.Iterator[T]: An iterator which iterates over all the nodes
        """
        pass

    @abstractmethod
    def op(self, *args, **kwargs) -> typing.Any:
        """
        """
        pass


def get_arg(arg, is_variable: bool=False) -> typing.Tuple[typing.Any, bool]:
    """Get the value for an arg

    Args:
        arg : The arg to retrieve
        is_variable (bool, optional): If the argument is variable. Defaults to False.

    Returns:
        typing.Tuple[typing.Any, bool]: The argument, whether it is a variable
    """

    if isinstance(arg, Output):
        return arg.value, False or is_variable
    elif isinstance(arg, T):
        return arg, True
    return arg, False or is_variable


class Output(T):
    """An output transmission defines the result of an
    operation in a node
    """

    def __init__(self, name: str, value, node: Node=None):
        """Create an output transmission

        Args:
            name (str): The name of the output
            value: The value of the output
            node (Node, optional): The node used to generate the output. Defaults to None.
        """
        super().__init__(name)
        self._value = value
        self._node = node

    @property
    def value(self) -> typing.Any:
        """
        Returns:
            typing.Any: The value for the output
        """
        return self._value
    
    @property
    def is_output(self) -> bool:
        return True

    @property
    def node(self) -> 'Node':
        """
        Returns:
            Node: The node that generated the output
        """
        return self._node

    # TODO: use "deep copy?"
    def clone(self) -> 'Output':
        """Clone the output

        Returns:
            Output: _description_
        """
        return Output(
            self.name, self._value
        )

    def __call__(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, deep: bool=True) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}
        stored[self.id] = self._value
        return self._value

    @property
    def incoming(self) -> typing.List['Incoming']:
        return []


class Process(T):
    """
    """

    def __init__(self, node: 'Node', args: typing.List=None, kwargs: typing.Dict=None):
        """Wraps a node in a graph. Each arg and kwargs will be used to define the graph
        # i feel the easy approach 

        Args:
            node (Node): Node to use 
            args (typing.List): The args to pass to the node
            kwargs (typing.Dict): The kwargs to pass to the node
        """
        super().__init__(node.name)
        self._node = node
        self._args: typing.List[T] = args or []
        self._kwargs: typing.Dict[str, T] = kwargs or {}

    @property
    def value(self) -> typing.Any:
        """

        Returns:
            typing.Any: The output of the process
        """
        return self.__call__()
    
    def clone(self) -> 'Process':
        """
        Returns:
            Process: The cloned process
        """
        args = [arg.clone() if isinstance(arg, T) else arg for arg in self._args]
        kwargs = {k: arg.clone() if isinstance(arg, T) else arg for k, arg in self._kwargs.items()}
        return Process(
            self._node, args, kwargs
        )

    @property
    def incoming(self) -> typing.List[Incoming]:
        incoming = []
        for i, arg in enumerate(self._args):
            incoming.append(Incoming(arg, i))
        for k, arg in self._kwargs.items():
            incoming.append(Incoming(arg, k))

        return incoming

    def __call__(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, deep: bool=True) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}

        args, kwargs = Incoming.prepare_args(self.incoming, by, stored)

        if deep:
            args = [arg(by, stored) if isinstance(arg, T) else arg for arg in args]
            kwargs = {
                k: arg(by, stored) if isinstance(arg, T) else arg for k, arg in kwargs.items()
            }
        
        result = stored[self.id] = self._node(*args, **kwargs)
        return result
        # result = stored[self._name][1]
        # if isinstance(result, Output):
        #     result = result.value
        # # TODO: I may need a different kind of instance
        # # because it is possible that a tuple is returned in some cases
        # elif isinstance(result, typing.Tuple):
        #     result = tuple(r_i.value if isinstance(r_i, Output) else r_i for r_i in result)
        # return result


class TIdx(T):
    """
    """

    def __init__(self, t: T, idx: int):
        """Wraps a node in a graph. Each arg and kwargs will be used to define the graph
        # i feel the easy approach 

        Args:
            node (Node): Node to use 
            args (typing.List): The args to pass to the node
            kwargs (typing.Dict): The kwargs to pass to the node
        """
        super().__init__(t.name + '_' + str(idx))
        self._idx = idx
        self._t = t

    @property
    def value(self) -> typing.Any:
        """

        Returns:
            typing.Any: The output of the process
        """
        return self.__call__()
    
    def clone(self) -> 'TIdx':
        """
        Returns:
            Process: The cloned process
        """
        # TODO: what if the underlying process is cloned
        return TIdx(
            self._t, self._idx
        )

    @property
    def incoming(self) -> typing.List[Incoming]:
        return [Incoming(self._t, 0)]

    @property
    def is_output(self) -> bool:
        return self._t.is_output

    def __call__(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, deep: bool=True) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}
        if deep:
            self._t(by, stored, True)
        result = stored[self.id] = stored[self._t.id][self._idx]
        return result


class Out(object):

    def __init__(self, outputs: typing.Union[
        typing.Tuple[T, int], T, 
        typing.List[typing.Union[typing.Tuple[T, int], T]]]):

        self._singular = not isinstance(outputs, typing.List)
        self._outputs = [outputs] if self._singular else outputs
        self._output_ts = []
        for output in self._outputs:
            if isinstance(output, typing.Tuple):
                self._output_ts.append(output[0])
            else:
                self._output_ts.append(output)

    @property
    def ts(self) -> typing.List[T]:
        return self._output_ts
    
    def idx(self, output: typing.Union[typing.Tuple[T, int], T], stored: typing.Dict) -> typing.Any:

        if isinstance(output, typing.Tuple):
            return stored[output[0].id][output[1]]
        return stored[output.id]
    
    def __call__(self, stored: typing.Dict[str, typing.Any]):

        result = tuple(
            self.idx(output, stored) for output in self._outputs
        )
        if self._singular:
            return result[0]
        return result


# TODO: See if this is necessary still
def to_by(trans: typing.List[Var], args: typing.List[str], kwargs: typing.Dict[str, typing.Any]) -> typing.Dict:
    """Convert the list of Vars to a 'by' dictionary

    Args:
        trans (typing.List[Var]): List of Variables
        args (typing.List[str]): The args for the function
        kwargs (typing.Dict[str, typing.Any]): The Kwargs to the function

    Raises:
        ValueError: If the key-value argument is duplicated

    Returns:
        typing.Dict: the By Dictionary
    """

    i = 0
    by = {}
    for t, arg in zip(trans, args):
        by[t] = arg
        i += 1
    for k, arg in kwargs.items():
        if k in by:
            raise ValueError(f'Key-value argument {k} has already been defined')
        by[k] = arg
    return by
