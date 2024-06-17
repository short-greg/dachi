# 1st party
import typing
from abc import abstractmethod
from typing import get_type_hints
from typing_extensions import Self
import inspect
import json
from io import StringIO
import csv

# 3rd party
import pydantic
from pydantic import Field


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

