# 1st party
from dataclasses import dataclass, field
import typing
from abc import abstractmethod
from io import StringIO


# 3rd party
import pandas as pd
import pydantic
from pydantic import Field, BaseModel

# local
from ..store import Struct, Str, Message, Chat

# https://www.city.amagasaki.hyogo.jp/map/1000379/1000416/1000814.html

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


class Styled(pydantic.BaseModel):

    data: Struct
    style: Style

    def forward(self, **kwargs):
        pass


class Context(pydantic.BaseModel):

    # contains structs
    # how to update structs with the inputs
    def forward(self, inputs, outputs) -> 'Context':
        # 1) add name fields
        # 
        # have to implement it something like this
        # need to know the names of the inputs, however
        # I can add the annotation
        pass
        # update all of the 
        # result = {}
        # for k, field in self.schema():

        #     attr = getattr(self, field)
        #     if isinstance(attr, Struct):
        #         result[k] = attr.forward()
        # return self.__class__(**result)


class Op(pydantic.BaseModel):
    
    name: Str
    inputs: typing.List[str] = Field(default_factory=list)
    descr: Str = field(default='')

    def forward(self, *inputs: typing.Union['Var', 'Struct']) -> 'Var':
        
        return Var(
            self, inputs
        )


# How about this?


# Have to specify the names of inputs here
# they are contained in the annotation

class Func(pydantic.BaseModel):

    name: str
    doc: str
    signature: str
    code: str
    inputs: typing.List[Struct]
    outputs: typing.List['Output']


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
        self, producer, name: str, incoming: typing.Union['Var', typing.Tuple['Var']]

    ):
        """

        Args:
            producer : 
            incoming (typing.Union[Var, typing.Tuple[Var]], optional): _description_. Defaults to None.
        """
        self.producer = producer
        self.incoming = incoming
        self.name = name

    def detach(self) -> 'Var':

        return Var(self.producer)


class Output(typing.Generic[S]):

    def __init__(
        self, incoming: typing.Union['Var', typing.Tuple['Var']],
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
