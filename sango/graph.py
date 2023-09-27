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
        pass


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
        pass


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
            return
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

    def __init__(self, name: str, dtype: str, default: typing.Union[typing.Any, typing.Callable[[], typing.Any]]=None):

        self.name = name
        self.dtype = dtype
        self.default = default


class Fields(object):

    def __init__(self, fields: typing.List[Field]):

        self.fields = fields


class Node(ABC):

    outputs = Fields([])

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
        self, name: str, inputs: typing.List[Var],
        outputs: typing.List[typing.Tuple[T, str]]
    ):
        super().__init__(name)
        
        self._inputs = inputs
        self._outputs = outputs

    def op(self, *args, **kwargs) -> typing.Any:
        
        # convert inputs / outputs
        # get the "by"

        # by = self._inputs.to_by(args, kwargs)
        by = {}
        stored = {}
        return tuple(
            self.outputs.get(t.probe(stored, by), k)
            for t, k in self._outputs
        )
        

# I need a way to spawn as well


class AppendStr(Node):

    def op(self, x: str="hi") -> T:
        return x + "_result"

t = Node(x=Var('x', str))


"""
I can define the names of the 
I can probe the tran


# Transmission contains data, transmission contains a link to the incoming node

"""
