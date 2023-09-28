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


class Const(T):

    def __init__(self, name: str, value):
        super().__init__(name)
        self._value = value

    @property
    def value(self) -> typing.Any:
        return self._value

    def probe(self, by: typing.Dict['Var', typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):
        return self._value


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
        #     if isinstance(arg, Process):
        #         arg._probe_helper(stored, by)
        #         arg = arg[arg._name]
        #     elif isinstance(arg, Var):
        #         arg = arg.validate(by)
        #     args.append(arg)
        # for k, arg in self._kwargs.items():
        #     if isinstance(arg, Process):
        #         arg.probe(by)
        #         arg = arg[arg._name]
        #     elif isinstance(arg, Var):
                
        #         arg = arg.validate(by)
        #     kwargs[k] = arg

    # def probe(self, by: typing.Dict[Var, typing.Any]=None, stored: typing.Dict[str, typing.Any]=None):
        
    #     by = by or {}
    #     stored = stored or {}

    #     return self._probe_helper(by, stored)


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


class TList(object):

    def __init__(self, trans: typing.List[typing.Union[typing.Tuple[T, str], T]]):

        self._trans: typing.List[T, str] = []
        for t in trans:
            if isinstance(t, T):
                pass
            else:
                pass

    @property
    def fields(self) -> 'FieldList':

        return [
            Field(t.name, t.dtype, t.value) for t in self._trans
        ]

    def probe(self, by: typing.Dict[Var, str], stored: typing.List[str]):
        
        results = []
        for t, key in self._trans:
            result = t.probe(by, stored)
            if key is not None:
                result = [result[key]]
            results.extend(result)
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


# TODO: Think of how to implement this
class Adapt(Node):

    def __init__(
        self, name: str, inputs: typing.List[Var], outputs: TList
    ):
        super().__init__(name)
        self._inputs = inputs
        self._outputs = outputs
    
    @property
    def outputs(self) -> FieldList:
        return self._outputs.fields

    def op(self, *args, **kwargs) -> typing.Any:
        
        by = to_by(self._inputs, args, kwargs)
        return self._outputs.probe(by)


class AppendStr(Node):

    def op(self, x: str="hi") -> T:
        return x + "_result"

t = Node(x=Var('x', str))


"""
I can define the names of the 
I can probe the tran


# Transmission contains data, transmission contains a link to the incoming node

"""
