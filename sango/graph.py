from abc import abstractmethod, ABC
import typing


class T(ABC):

    def __init__(self, name: str):
        self._name = name

    @property
    def value(self) -> typing.Any:
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


class Const(T):

    def __init__(self, name: str, value):
        super().__init__(name)
        self._value = value

    @property
    def value(self) -> typing.Any:
        return self._value

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):
        return self._value

    # TODO: use "deep copy?"
    def clone(self) -> 'Const':
        return Const(
            self.name, self._value
        )


class Var(T):

    def __init__(self, name: str, dtype: str=None, default=None):
        super().__init__(name)
        self.dtype = dtype
        self._default = default

    @property
    def value(self) -> typing.Any:
        return self._default

    def set(self, default):
        self._default = default

    def validate(self, kv: typing.Dict['Var', typing.Any]):

        try:
            val = kv[self]
        except KeyError:
            return self._default
        if self.dtype is None or not isinstance(val, self.dtype):
            raise ValueError(f'Value must be of dtype {self.dtype}')
        return val        

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):
        return self.validate(by)

    def clone(self) -> 'Var':
        return Var(
            self.name, self.dtype, self._default
        )


class Process(T):

    def __init__(self, node: 'Node', args: typing.List, kwargs: typing.Dict):
        
        super().__init__(node.name)
        self._node = node
        self._args: typing.List[T] = args
        self._kwargs: typing.Dict[str, T] = kwargs

    @property
    def value(self) -> typing.Any:
        return self.probe(self.name)
    
    def probe(self, by: typing.Dict[Var, typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):

        if self._name in stored:
            return stored[self._name]
        args = []
        kwargs = {}
        for arg in self._args:
            args[k] = arg.probe(by, stored)
        for k, arg in self._kwargs.items():
            kwargs[k] = arg.probe(by, stored)
        stored[self._name] = self._node(*args, **kwargs)
        return stored[self._name]

    # TODO: Use deepcopy?
    def clone(self) -> 'Process':
        return Process(
            self._node, self._args, self._kwargs
        )


def get_arg(arg, is_variable: bool=False):

    if isinstance(arg, Var) or isinstance(arg, Process):
        return arg, True
    elif isinstance(arg, Const):
        return arg.value, False or is_variable
    return arg, False or is_variable


class Field(object):

    def __init__(self, name: str, dtype: str=None, default: typing.Union[typing.Any, typing.Callable[[], typing.Any]]=None):

        self.name = name
        self.dtype = dtype
        self.default = default

    @classmethod
    def factory(self, f) -> 'Field':
        if isinstance(f, Field):
            return Field(f.name, f.dtype, f.default)
        if isinstance(f, str):
            return Field(f)
        if isinstance(f, typing.Iterable):
            return Field(*f)
        raise ValueError(f'Argument f must be of type Field, string or tuple')


class FieldList(object):

    def __init__(self, fields: typing.List[Field]):

        self.fields = [Field.factory(field) for field in fields]

    def index(self, name: str) -> int:

        for i, field in enumerate(self.fields):
            if field.name == name:
                return i
        raise ValueError(f'No field named {name} in the Field List')


def probe_ts(ts: typing.List[typing.Union[T, typing.Tuple[T, int]]], by: typing.Dict[Var, str], stored: typing.List[str]):
    
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


class Node(ABC):

    outputs = FieldList([])

    def __init__(self, name: str):
        
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def op(self) -> typing.Any:
        pass

    def __call__(self, *args, **kwargs) -> typing.Union[typing.Any, typing.Tuple[typing.Any]]:
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

        result = self.op(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(Const(self._name, result_i) for result_i in result)
        return Const(self._name, result)
    
    @abstractmethod
    def clone(self) -> 'Node':
        pass


class Tako(Node):
    
    @abstractmethod
    def traverse(self, **kwargs) -> typing.Iterator[T]:
        pass



# TODO: FINISH!
class Adapt(Tako):

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
    
    def traverse(self, **kwargs) -> typing.Iterator[T]:
        # create a traverse method
        yield None


class AppendStr(Node):

    def op(self, x: str="hi") -> T:
        return x + "_result"

t = Node(x=Var('x', str))


"""
I can define the names of the 
I can probe the tran


# Transmission contains data, transmission contains a link to the incoming node

"""
