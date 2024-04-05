# 1st party
from dataclasses import dataclass, field
import typing
from abc import abstractmethod
from io import StringIO

# 3rd party
import pandas as pd
import pydantic

# local
from ..store import Struct, Str, Message, Chat



T = typing.TypeVar('T', bound=Struct)
S = typing.TypeVar('S', bound=Struct)


class InstructChat(Chat):

    def __init__(self, structs: typing.List[Message] = None, instructor: str='system'):
        super().__init__(structs)
        self.instructor = instructor

    def chat(self) -> Chat:

        return Chat(

            s for s in self._structs if s.role != self.instructor
        )


# TODO: FINISH THIS
class Style(typing.Generic[S]):

    @abstractmethod
    def write(self, text: str):
        pass
    
    @abstractmethod
    def read(self, text: str) -> S:
        pass


class Context(pydantic.BaseModel):

    # contains structs
    # how to update structs with the inputs

    def __call__(self, **kwargs):
        pass


class Op(pydantic.BaseModel):
    
    name: Str
    inputs: typing.List[str] = None
    descr: Str

    def forward(self, *inputs: typing.Union['Var', 'Struct']) -> 'Var':
        
        return Var(
            self, inputs
        )


def output(*inputs, style: Style[S]) -> 'Output[S]':

    return Output[S](
        inputs, style
    )


class _op:

    def __getattr__(self, key: str):

        name = key
        def _op_creator(descr: str) -> Op:
            return Op(name=name, descr=descr)
        return _op_creator


op = _op()        


class Var(object):

    def __init__(
        self, producer, incoming: typing.Union['Var', typing.Tuple['Var']]=None
    ):
        self.producer = producer
        self.incoming = incoming

    def detach(self) -> 'Var':

        return Var(self.producer)


class Output(typing.Generic[S]):

    def __init__(
        self, incoming: typing.Union['Var', typing.Tuple['Var']]=None,
        style: Style=None
    ):
        self.style = style
        self.incoming = incoming

    def detach(self) -> 'Output':

        return Output(self.producer)


class Param(typing.Generic[S]):

    def __init__(
        self, struct: S, style: Style
    ):
        """

        Args:
            struct (S): 
            style (Style): 
        """
        self.struct = struct
        self.style = style

    def detach(self) -> 'Param':

        return Param(self.struct)


def instrmethod(f):

    def _(self, *args, **kwargs):

        output = f(self, *args, **kwargs)

    return _


def instr(f):

    def _(*args, **kwargs):

        # 
        output = f(*args, **kwargs)

    return _



# class Context(object):
    
#     def __init__(self, struct: Struct, style: Style=None):

#         self.struct = struct
#         self.style = style

#     def __call__(self, style_override: Style=None):

#         if style_override is None:
#             return self.style.write(self.struct)
        
#         return style_override.write(self.struct)



# C = typing.TypeVar('C', bound=Struct)


# class ContextF(typing.Generic[C]):
    
#     def __init__(self, structf: typing.Callable[[], C], style: Style=None):

#         self.structf = structf
#         self.style = style

#     def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> Context:
#         return Context(
#             self.structf(*args, **kwargs), self.style
#         )
