# 1st party
from dataclasses import field
import typing

# 3rd party
import pydantic

# local
from .._core._struct import Struct, Str, ValidateStrMixin


class Role(Struct):

    name: str
    descr: str = field(default_factory=lambda: '')


class Text(Struct):

    text: str = ''
    descr: str = field(default_factory=lambda: '')


class Body(Struct):

    sep_before: str='-------'
    sep_after: str='-------'

    def fill(self, struct: Struct):
        pass


class Var(pydantic.BaseModel):

    pass


class Op(pydantic.BaseModel, ValidateStrMixin):

    name: str
    inputs: typing.List[Struct]
    descr: str = ''

    def forward(self, var: 'Var') -> 'Var':
        return Var(
            
        )
