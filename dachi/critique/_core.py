from abc import ABC, abstractmethod
import typing
import json

import pydantic

from .._core import (
    Struct, Description, Module, 
    escape_curly_braces,
    get_str_variables
)
from .. import converse


class Sample(Struct):

    data: typing.Dict[str, Struct]

    def render(self) -> str:

        return escape_curly_braces(
            self.data
        )


class Batch(Struct):

    data: typing.List[Sample]

    def render(self) -> str:
        return escape_curly_braces(
            self.data
        )


Data = typing.Union[Batch, Sample]


class Assessment(Struct):

    name: str
    description: str
    result: str


class Evaluate(Description, Module):

    name: str
    how: str

    @property
    @abstractmethod
    def var_names(cls) -> typing.Set[str]:
        pass

    @pydantic.field_validator('how', mode='before')
    def validate_names_types_data(cls, values):
    
        variables = set(get_str_variables())
        
        if variables != cls.var_names:
            raise ValueError(
                "The description must have these variable "
                f"names {cls.var_names}"
            )

        return values

    @abstractmethod
    def out_format(self) -> typing.Dict:
        pass

    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def additional(self) -> typing.Dict:
        return {}

    def render(self) -> str:
        return escape_curly_braces(self.out_format())
    
    @abstractmethod
    def forward(self, y: Data, t: Data) -> typing.Any:
        pass


class Supervised(Evaluate):

    def out_format(self) -> typing.Dict:
        
        return {
            self.name: f'Evaluate the input y according to '
                       f'this regularzation. {self.regularization}'
        }
    
    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def forward(self, y: Data, t: Data) -> typing.Any:
        
        rendered = self.render()
        data = y.merge(t)
        return rendered.format(data=data.render())


class Quality(Evaluate):

    def out_format(self) -> typing.Dict:
        
        return {
            self.name: f'Evaluate the input y according to '
                       f'this regularzation. {self.regularization}'
        }
    
    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def forward(self, y: Data, t: Data=None) -> typing.Any:
        
        rendered = self.render()
        data = y.merge(t)
        return rendered.format(data=data.render())


class Style(Evaluate):

    def render(self) -> str:
        return super().render()

    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def additional(self) -> typing.Dict:
        return {}

    def forward(self, y: Data, t: Data=None) -> typing.Any:
        
        rendered = self.render()
        data = y.merge(t)
        return rendered.format(data=data.render())


class Critic(Module, ABC):

    @abstractmethod
    def forward(self, x: Data, t: Data=None) -> typing.Any:
        pass


class LLMCritic(Module, ABC):

    def __init__(self, llm, evaluation: Evaluate):

        self.llm = llm
        self.evaluation = evaluation
    
    def forward(self, y: Data, t: Data=None) -> Assessment:
        
        instructions = self.evaluation(
            y, t
        )
        result = self.llm(instructions)
        return Assessment(**json.loads(result))


class Evaluate(Struct):
    """Evaluate the output of an AI model
    Typically with some kind of language output
    """
    results: typing.List[typing.Dict[str, Assessment]]

