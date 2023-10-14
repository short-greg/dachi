from abc import abstractmethod, ABC
import typing


class T(ABC):
    """Define a transmission to pass through the graph"""

    def __init__(self, name: str):
        """Create a transition

        Args:
            name (str): Name of the transmission
        """
        self._name = name

    @property
    def value(self) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        pass

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):
        pass

    @abstractmethod
    def clone(self) -> 'T':
        pass


class Field(object):
    """
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
        """Create a field based on the value f

        Args:
            f : The values for field

        Raises:
            ValueError: If the value for the field is invalid

        Returns:
            Field: The resulting field
        """
        if isinstance(f, Field):
            return Field(f.name, f.dtype, f.default)
        if isinstance(f, str):
            return Field(f)
        if isinstance(f, typing.Iterable):
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


class Node(ABC):
    """A node in the graph defines the operation to execute. 
    """

    # Define the ouptuts for the operation
    outputs = FieldList([])

    def __init__(self, name: str):
        """Create a node in the graph

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

    def op(self) -> typing.Any:
        """The operation for the node. 

        Returns:
            typing.Any: The result of the node
        """
        pass

    def __call__(self, *args, **kwargs) -> typing.Union[T, typing.Tuple[T], typing.Any, typing.Tuple[typing.Any]]:
        """Use for transmissions. This wraps the operation

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
        return self._value
    
    @property
    def node(self) -> 'Node':
        return self._node

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None) -> typing.Any:
        """

        Args:
            by (typing.Dict[Var, typing.Any], optional): The inputs used to retrieve. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): The currently stored values. Defaults to None.

        Returns:
            typing.Any: The result of the operation
        """
        return self._value

    # TODO: use "deep copy?"
    def clone(self) -> 'Output':
        return Output(
            self.name, self._value
        )

class F(object):

    def __init__(self, f: typing.Callable[[], typing.Any]):
        self.f = f

    @property
    def value(self):
        return self.f()
 

class Var(T):

    def __init__(self, name: str, dtype: str=None, default: typing.Union[typing.Any, F]=None):
        """

        Args:
            name (str): Name of the variable
            dtype (str, optional): The type of the variable. Defaults to None.
            default (_type_, optional): Default variable for the variable. Defaults to None.
        """
        super().__init__(name)
        self.dtype = dtype
        self._default = default

    @property
    def value(self) -> typing.Any:
        if isinstance(self._default, F):
            return self._default.value
        return self._default

    def set(self, default):
        self._default = default

    def validate(self, by: typing.Dict['Var', typing.Any]):

        try:
            val = by[self]
        except KeyError:
            return self._default
        if self.dtype is not None and not isinstance(val, self.dtype):
            raise ValueError(f'Value must be of dtype {self.dtype}')
        return val        

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None) -> typing.Any:
        """

        Args:
            by (typing.Dict[Var, typing.Any], optional): . Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): . Defaults to None.

        Returns:
            typing.Any: The result of the probe
        """
        return self.validate(by)

    def clone(self) -> 'Var':
        return Var(
            self.name, self.dtype, self._default
        )


class Process(T):
    """Define a graph of nodes. 
    """

    def __init__(self, node: 'Node', args: typing.List=None, kwargs: typing.Dict=None):
        """Wraps a node in a graph. Each arg and kwargs will be used to define teh graph

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
            typing.Any: _description_
        """
        return self.probe()
    
    def probe(self, by: typing.Dict[Var, typing.Any]=None, stored: typing.Dict[str, typing.Any]=None) -> typing.Any:
        """

        Args:
            by (typing.Dict[Var, typing.Any], optional): _description_. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): _description_. Defaults to None.

        Returns:
            typing.Any: The result of the probe
        """
        by = by or {}
        stored = stored or {}
        if self._name in stored:
            return stored[self._name]
        args = []
        kwargs = {}
        for arg in self._args:
            if isinstance(arg, T):
                arg = arg.probe(by, stored)
            args.append(arg)
        for k, arg in self._kwargs.items():
            if isinstance(arg, T):
                arg = arg.probe(by, stored)
            kwargs[k] = arg
        stored[self._name] = self._node(*args, **kwargs)
        result = stored[self._name]
        if isinstance(result, typing.Iterable):
            return tuple(r_i.value for r_i in result)
        return result.value

    def clone(self) -> 'Process':
        """
        Returns:
            Process: The cloned process
        """
        return Process(
            self._node, self._args, self._kwargs
        )


def probe_ts(ts: typing.List[typing.Union[T, typing.Tuple[T, int]]], by: typing.Dict[Var, str]=None, stored: typing.Dict[str, typing.Any]=None):
    
    by = by or {}
    stored = stored or {}

    results = []
    for t in ts:
        if isinstance(t, tuple):
            t, idx = t
        else:
            idx = None
        result = t.probe(by, stored)
        if isinstance(result, typing.Iterable):
            if idx is None:
                raise ValueError(f'Index must be specified for {t.name}')
            result = result[idx]
        results.append(result)

    return tuple(results)


def get_arg(arg, is_variable: bool=False):

    if isinstance(arg, Var) or isinstance(arg, Process):
        return arg, True
    elif isinstance(arg, Output):
        return arg.value, False or is_variable
    return arg, False or is_variable


def to_by(trans: typing.List[Var], args: typing.List[str], kwargs: typing.Dict[str, typing.Any]):
    
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


# TODO: Figure out how to implement
def traverse():
    pass


class Tako(Node):
    
    @abstractmethod
    def traverse(self, **kwargs) -> typing.Iterator[T]:
        pass

    @abstractmethod
    def op(self, *args, **kwargs) -> typing.Any:
        pass


class Adapter(Node):

    def __init__(
        self, name: str, inputs: typing.List[Var], outputs: typing.List[typing.Union[typing.Tuple[T, int], T]]
    ):
        super().__init__(name)
        self._inputs = inputs
        self._outputs = outputs
    
    @property
    def outputs(self) -> FieldList:
        return self._outputs.fields

    def op(self, *args, **kwargs) -> typing.Any:
        
        by = to_by(self._inputs, args, kwargs)
        return probe_ts(self._outputs, by)


"""
I can define the names of the 
I can probe the tran


# Transmission contains data, transmission contains a link to the incoming node

"""
