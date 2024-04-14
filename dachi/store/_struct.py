import pandas as pd
import typing
from pydantic import Field
import pydantic
from abc import abstractmethod

from typing import get_type_hints
import inspect
from abc import abstractmethod

import csv
import pandas as pd
from io import StringIO
from typing_extensions import Self
import json


class TextMixin(object):

    @abstractmethod
    def to_text(self) -> str:
        pass


def to_text(value):

    if isinstance(value, TextMixin):
        return value.to_text()
    return str(value)


class Str(pydantic.BaseModel, TextMixin):

    text: str
    vars: typing.List[str] = Field(default_factory=list)

    def forward(self, **kwargs):

        format = {}
        remaining_vars = []
        for var in self.vars:
            if var in kwargs:
                format[var] = to_text(kwargs[var])
            else:
                format[var] = '{' + var + '}' 
                remaining_vars.append(var)
        return Str(
            text=self.text.format(**format),
            vars=remaining_vars
        )
    
    def to_text(self):

        return self.text
    
    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    
    template = {}
    for name, field_type in get_type_hints(model_cls).items():
        
        if inspect.isclass(field_type) and issubclass(field_type, pydantic.BaseModel):
            template[name] = model_template(field_type)
        else:
            template[name] = {
                "is_required": model_cls.model_fields[name].is_required(),
                "type": field_type
            }
    return template


class ValidateStrMixin:

    @pydantic.field_validator('*', mode='before')
    def convert_to_string_template(cls, v, info: pydantic.ValidationInfo):
    
        outer_type = cls.model_fields[info.field_name].annotation
        if (inspect.isclass(outer_type) and issubclass(outer_type, Str)) and not isinstance(v, Str) and not isinstance(v, typing.Dict):
            return Str(text=v)
        return v


class Struct(pydantic.BaseModel, TextMixin, ValidateStrMixin):

    model_config = pydantic.ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    

    def forward(self, **kwargs) -> 'Struct':
        
        new_args = {}
        for field in self.model_fields:
            attr = getattr(self, field)
            if isinstance(attr, Struct):
                new_args[field] = attr(**kwargs)
            elif isinstance(attr, Str):
                new_args[field] = attr(**kwargs)
            else:
                new_args[field] = attr
        return self.__class__(**new_args)

    @classmethod
    def template(cls) -> str:
        return model_template(cls)

    def __call__(self, **kwargs) -> 'Struct':
        return self.forward(**kwargs)
    
    def to_text(self) -> str:
        return str(self.model_dump())
    
    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(
            **json.loads(text)
        )


T = typing.TypeVar('T', bound=Struct)

class Message(Struct):

    role: Str
    text: Str



class Doc(Struct):

    name: Str
    text: Str


#     # {role}: {text}


class StructList(Struct, typing.Generic[T]):

    structs: typing.List[T]


class Chat(Struct):

    messages: typing.List[Message]

    def filter(self, roles: typing.Iterable[str]) -> 'Chat[Message]':

        roles = set(roles)
        
        return Chat(
            s for s in self._structs if s.role in roles
        )
