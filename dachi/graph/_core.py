from abc import abstractmethod, ABC
import typing


def _is_function(f):

    f_type = type(f)
    return f_type == type(_is_function) or f_type == type(hasattr)


class T(ABC):
    """Base transmission class"""

    def __init__(self, name: str):
        """Create a transmission from one node to another

        Args:
            name (str): Name of the transmission
        """
        self._name = name

    @property
    def value(self) -> typing.Any:
        """
        Returns:
            typing.Any: The output of the transmission
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the transmission
        """
        return self._name

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None) -> typing.Any:
        """Retrieve the value of the transmission

        Args:
            by (typing.Dict[Var, typing.Any], optional): Storage of the values to use for the vars. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): Storage of the outputs of the nodes in the graph. Defaults to None.

        Returns:
            Any: The output of the transmission
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}

        for _, result in self.traverse(by, stored, True):
            pass

        result = stored[self._name][1]
        if isinstance(result, Output):
            result = result.value
        # TODO: I may need a different kind of instance
        # because it is possible that a tuple is returned in some cases
        elif isinstance(result, typing.Tuple):
            result = tuple(r_i.value if isinstance(r_i, Output) else r_i for r_i in result)
        return result

    @abstractmethod
    def clone(self) -> 'T':
        pass

    @abstractmethod
    def retrieve(self, key: str) -> 'T':
        pass

    @abstractmethod
    def traverse(
        self, by: typing.Dict['Var', typing.Any]=None, 
        stored: typing.Dict[str, typing.Any]=None, 
        evaluate: bool=False
    ) -> typing.Iterator[typing.Tuple['T', typing.Any]]:
        """
        Args:
            by (typing.Dict[Var, typing.Any], optional): The values to probe vars by. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): The currently stored values. Defaults to None.
            evaluate (bool, optional)
        Returns:
            typing.Any: The result of the probe
        """
        pass


class F(object):

    def __init__(self, f: typing.Callable[[], typing.Any], *args, **kwargs):

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


class NodeMethod(Node):

    def __init__(self, obj, f, inputs: typing.List[Field], outputs: typing.List[Field]):

        super().__init__(f'{obj.__class__.__name__}_{f.__name__}')
        self._obj = obj
        self._f = f
        self._inputs = inputs
        self._outputs = outputs

    def op(self, *args, **kwargs) -> typing.Any:
        
        return self._f(self._obj, *args, **kwargs)


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


def nodemethod(inputs: typing.List[Field], outputs: typing.List[Field]):
    """Decorator that transforms a method to work as a node. To get the Node object you 
    must pass in get_node=True. To link the transmission pass in link=True

    Args:
        inputs (typing.List[Field]): Definition of the inputs to the Node
        outputs (typing.List[Field]): Definition of the outputs to the Node
    """

    def _out(f):
        node = None

        def _in(self, *args, link: bool=False, get_node: bool=False, **kwargs):
            nonlocal node

            if node is None:
                node = NodeMethod(
                    self, f, inputs, outputs
                )
            if link:
                if get_node is True:
                    raise ValueError('Cannot link if GetNode is true')
                return node.link(*args, **kwargs)
            if get_node:
                if len(args) != 0 or len(kwargs) != 0:
                    raise ValueError('Must not pass in args or kwargs if getting the node.')
                return node
            
            return node(*args, **kwargs)
        _in.node_method = True
        return _in
    return _out


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

    def retrieve(self, key: str) -> 'T':
        """
        Args:
            key (str): The key to retrieve

        Returns:
            T: Self if self == the Key
        """
        if self._name == key:
            return self

    def traverse(
        self, by: typing.Dict['Var', typing.Any]=None, 
        stored: typing.Dict[str, typing.Any]=None, 
        evaluate: bool=False
    ) -> typing.Iterator[typing.Union[typing.Tuple['T', typing.Any], T]]:
        """
        Args:
            by (typing.Dict[Var, typing.Any], optional): The values to probe vars by. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): The currently stored values. Defaults to None.
            evaluate (bool, optional)
        Returns:
            typing.Any: The result of the probe
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}
        if evaluate:
            result = self, self._value
        else: 
            result = self
        stored[self._name] = result
        yield result


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

    def retrieve(self, key: str) -> 'T':
        if self._name == key:
            return self 

    def traverse(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, evaluate: bool=False) -> typing.Iterator[typing.Tuple['T', typing.Any]]:
        """
        Args:
            by (typing.Dict[Var, typing.Any], optional): The values to probe vars by. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): The currently stored values. Defaults to None.
            evaluate (bool, optional): Whether to evaluate the result and return it
        Returns:
            typing.Any: The result of the probe
        """
        stored = stored if stored is not None else {}
        by = by if by is not None else {}
        if evaluate:
            result = self, self.validate(by)
        else:
            result = self
        stored[self._name] = result
        yield result


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
        return self.probe()
    
    def traverse(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None, evaluate: bool=False) -> typing.Iterator[typing.Tuple['T', typing.Any]]:
        """

        Args:
            by (typing.Dict[Var, typing.Any], optional): _description_. Defaults to None.
            stored (typing.Dict[str, typing.Any], optional): _description_. Defaults to None.

        Returns:
            typing.Any: The result of the probe
        """

        stored = stored if stored is not None else {}
        by = by if by is not None else {}
        if self._name in stored:
            cur = stored[self._name]
            if cur == self and not evaluate or cur != self and evaluate:
                yield cur
                return
        args = []
        kwargs = {}
        for arg in self._args:
            if isinstance(arg, T):
                for cur_arg in arg.traverse(by, stored, evaluate=evaluate):
                    yield cur_arg
                
                arg = cur_arg if not evaluate else cur_arg[1] # arg.probe(by, stored)
            args.append(arg)
        for k, arg in self._kwargs.items():
            if isinstance(arg, T):
                # If T is asynchronous
                # 1) already "running"
                # 2) not running yet
                # 3) In either case, I do a "traverse_async" here
                # 4) then before I execute the node I need to collect the results.. 
                # ... I'll want to create a sandbox to demonstrate this first
                for cur_arg in arg.traverse(by, stored, evaluate=evaluate):
                    yield cur_arg
                arg = cur_arg if not evaluate else cur_arg[1] # arg.probe(by, stored)
                # arg = arg.probe(by, stored)
            kwargs[k] = arg
        if evaluate:
            # here is where I execute the node... 
            # there is a possibility that the node is 
            # async or streaming
            result = self, self._node(*args, **kwargs)
        else:
            result = self
        stored[self._name] = result
        yield result

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

    def retrieve(self, key: str) -> 'T':
        if self._name == key:
            return self
        for arg in self._args:
            if isinstance(arg, T):
                result = arg.retrieve(key)
                if result is not None:
                    return result

        for key, arg in self._kwargs.items():
            if isinstance(arg, T):
                result = arg.retrieve(key)
                if result is not None:
                    return result


def probe_ts(ts: typing.List[typing.Union[T, typing.Tuple[T, int]]], by: typing.Dict[Var, str]=None, stored: typing.Dict[str, typing.Any]=None):
    """Probe the output of transmissions

    Args:
        ts (typing.List[typing.Union[T, typing.Tuple[T, int]]]): The transmissions to query
        by (typing.Dict[Var, str], optional): The values to set for the variables. Defaults to None.
        stored (typing.Dict[str, typing.Any], optional): The stored outputs for the nodes. Defaults to None.

    Raises:
        ValueError: If an index is not specified for results that return multiple outputs

    Returns:
        _type_: _description_
    """

    stored = stored if stored is not None else {}
    by = by if by is not None else {}

    results = []
    for t in ts:
        if isinstance(t, typing.Tuple):
            t, idx = t
        else:
            idx = None
        result = t.probe(by, stored)
        to_extend = False
        if isinstance(result, typing.Tuple):
            if idx is None:
                raise ValueError(f'Index must be specified for {t.name}')
            if isinstance(idx, slice) or isinstance(idx, typing.Iterable):
                to_extend = True
            result = result[idx]
        if to_extend:
            results.extend(result)
        else: results.append(result)

    return tuple(results)


def traverse_ts(
    ts: typing.List[typing.Union[T, typing.Tuple[T, int]]], 
    by: typing.Dict[Var, str]=None, stored: typing.Dict[str, typing.Any]=None,
    evaluate: bool=False
):
    """Probe the output of transmissions

    Args:
        ts (typing.List[typing.Union[T, typing.Tuple[T, int]]]): The transmissions to query
        by (typing.Dict[Var, str], optional): The values to set for the variables. Defaults to None.
        stored (typing.Dict[str, typing.Any], optional): The stored outputs for the nodes. Defaults to None.

    Raises:
        ValueError: If an index is not specified for results that return multiple outputs

    Returns:
        the result
    """

    stored = stored if stored is not None else {}
    by = by if by is not None else {}

    results = []
    for t in ts:
        for result in t.traverse(by, stored, evaluate):
            yield result

    return tuple(results)


def get_arg(arg, is_variable: bool=False) -> typing.Tuple[typing.Any, bool]:
    """Get the value for an arg

    Args:
        arg : The arg to retrieve
        is_variable (bool, optional): If the argument is variable. Defaults to False.

    Returns:
        typing.Tuple[typing.Any, bool]: The argument, whether it is a variable
    """

    if isinstance(arg, Var) or isinstance(arg, Process):
        return arg, True
    elif isinstance(arg, Output):
        return arg.value, False or is_variable
    return arg, False or is_variable


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
