# 1st party
from abc import abstractmethod, ABC
import typing
from typing_extensions import Self

# 3rd party
import networkx as nx


class Src(ABC):

    @abstractmethod
    def incoming(self) -> typing.Iterator['T']:
        pass

    @abstractmethod
    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        pass

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        return self.forward(by)


class T(object):

    def __init__(
        self, val=None, src: Src=None,
        multi: bool=False
    ):
        self._val = val
        self._src = src
        self._multi = multi

    @property
    def val(self) -> typing.Any:
        return self.val

    @property
    def undefined(self) -> bool:

        return self._val is None

    def __getitem__(self, idx: int) -> 'T':
        
        if not self._multi:
            raise RuntimeError(
                'Object T does not have multiple objects'
            )
        else:
            val = None
        return T(
            val, IdxSrc(self, idx)
        )
    
    def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        if self._val is not None:
            return self._val

        if self in by:
            return by[self]
    
        if self._src is not None:
            for incoming in self._src.incoming():
                incoming.probe(by)
            by[self] = self.src(by)
            return by[self]
        
        raise RuntimeError('Val has not been defined and no source for t')

    def detach(self):
        return T(
            self._val, None, self._multi
        )


class Var(Src):
    
    def __init__(self, default=None, default_factory=None):

        self.default = default
        self.default_factory = default_factory

    def incoming(self) -> typing.Iterator[T]:

        # hack to ensure it is a generator
        if False:
            yield False
        
    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by.get[self]
        if self.default is not None:
            return self.default
        if self.default_factory is not None:
            return self.default_factory()
        
        raise RuntimeError('')


class IdxSrc(Src):

    def __init__(self, t: T, idx):

        self.t = t
        self.idx = idx

    def incoming(self) -> typing.Iterator['T']:
        yield self.t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by[self]
        return self.t.probe(by)[self.idx]


class Args(object):

    def __init__(self, *args, **kwargs):
        
        undefined = False

        for arg in args:
            
            if isinstance(arg, T):
                if arg.undefined:
                    undefined = True
                    break
        for k, arg in kwargs.items():
            
            if isinstance(arg, T):
                if arg.undefined:
                    undefined = True
                    break
        self._args = args
        self._undefined = undefined
        self._kwargs = kwargs
    
    @property
    def undefined(self) -> bool:
        return self._undefined
    
    @property
    def args(self) -> typing.List:
        return self._args
    
    @property
    def kwargs(self) -> typing.Dict:
        return self._kwargs
    
    def incoming(self) -> typing.Iterator['T']:

        for arg in self._args:
            if isinstance(arg, T):
                yield arg

        for k, arg in self._kwargs:
            if isinstance(arg, T):
                yield arg
    
    def forward(self, by: typing.Dict['T', typing.Any]) -> Self:

        args = []
        kwargs = {}
        for arg in self._args:
            if isinstance(arg, T) and arg in by:
                args.append(by[arg])
            else:
                args.append(arg)
            
        for k, arg in self._kwargs.items():
            if isinstance(arg, T) and arg in by:
                kwargs[k] = by[arg]
            else:
                kwargs[k] = arg
            
        return Args(*args, **kwargs)
    
    def __call__(self, by: typing.Dict['T', typing.Any]) -> Self:
        return self.forward(by)

# TODO: Handle multi

class FSrc(Src):

    def __init__(self, mod: 'Module', args: Args):

        super().__init__()
        self.mod = mod
        self.args = args

    def incoming(self) -> typing.Iterator['T']:
        
        for t in self.args.incoming():
            yield t

    def forward(self, by: typing.Dict[T, typing.Any]) -> typing.Any:
        
        if self in by:
            return by[self]
        
        args = self.args(by)
        return self.mod(*args.args, **args.args).val


class Module(ABC):

    def __init__(self, multi_out: bool=False):

        self._multi_out = multi_out

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        pass

    def __call__(self, *args, **kwargs) -> T:

        args = Args(*args, **kwargs)
        if not args.undefined:
            return T(
                self.forward(*args.f_args, **args.f_kwargs),
                FSrc(self, args), self._multi_out
            )
        return T(
            None, FSrc(self, args), self._multi_out
        )


class StreamSrc(Src):
    # use to nest streaming operations

    pass


class AsyncSrc(Src):
    # use to nest async operations


    pass

