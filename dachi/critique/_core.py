from abc import ABC, abstractmethod
import typing
import json
from ..converse import PromptModel, Message


import pydantic

from .._core import (
    Struct, Description, Module, 
    escape_curly_braces,
    get_str_variables,
    str_formatter
)


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


class EvaluatorBase(Description, Module):

    @abstractmethod
    def out_format(self) -> typing.Dict:
        pass

    def out_format_str(self) -> str:
        base_out = escape_curly_braces(
            self.out_format()
        )
        return f"""
        {
            0: {base_out},
            ...
            N: {base_out}
        }
        """

    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def additional(self) -> typing.Dict:
        return {}

    def additional_str(self) -> str:
        additional = self.additional()
        if len(additional) == 0:
            return 'None.'
        out_str = ""
        for k, v in additional.items():
            out_str += f'{k}: {escape_curly_braces(v)}\n\n'
        return out_str

    def render(self) -> str:
        rendered = """
        Evaluate each sample from 1 to N where N is the number of
        samples

        # Criteria

        {criteria}

        # 
        {data}

        Output with this format.

        {format}

        # Additional

        {additional}
        """
        return str_formatter(
            rendered, criteria=escape_curly_braces(self.criteria),
            format=escape_curly_braces(self.out_format_str()),
            additional=escape_curly_braces(self.additional())
        )
    
    @abstractmethod
    def forward(self, y: Data, t: Data) -> typing.Any:
        pass


class Evaluator(EvaluatorBase, Module):

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
    def out_format_str(self) -> typing.Dict:
        pass

    @abstractmethod
    def criteria(self) -> typing.Dict:
        pass

    def additional(self) -> typing.Dict:
        return {}

    def render(self) -> str:
        return escape_curly_braces(self.out_format_str())
    
    @abstractmethod
    def forward(self, y: Data, t: Data) -> typing.Any:
        pass


class CompositeEvaluate(EvaluatorBase):

    def __init__(self, evaluators: typing.List[Evaluator]):
        
        self.evaluators = evaluators

    def out_format_str(self) -> typing.Dict:
        
        format = {}
        for evaluator in self.evaluators:
            format.update(
                evaluator.out_format_str()
            )
        return format

    def criteria(self) -> typing.Dict:

        criteria = {}
        for evaluator in self.evaluators:
            criteria.update(
                evaluator.criteria()
            )
        return criteria

    def additional(self) -> typing.List[typing.List[str]]:

        additional = []
        for evaluator in self.evaluators:
            additional.append(
                *evaluator.additional()
            )
        return additional

    def render(self) -> str:

        return escape_curly_braces(
            self.out_format_str()
        )
    
    def forward(self, y: Data, t: Data=None) -> typing.Any:
        
        result = self.render()
        variables = get_str_variables(result)
        if 't' in variables and t is not None:
            return result.format(
                y=y, t=t
            )
        return result.format(y=y)


class Supervised(Evaluator):

    def out_format(self) -> typing.Dict:
        
        return {self.name: '<Evaluation>'}

    def criteria(self) -> typing.Dict:
        return {
            self.name: f'{self.name}: how well the output matches the target'
                       f'according to: {self.how}'
        }
    
    def render(self) -> str:
        return super().render()

    def forward(self, y: Data, t: Data) -> typing.Any:
        
        rendered = self.render()
        data = y.merge(t)
        return rendered.format(data=data.render())


class Quality(Evaluator):

    def out_format_str(self) -> typing.Dict:
        
        return {
            self.name: f'Evaluate the input y according to '
                       f'this regularzation. {self.regularization}'
        }
    
    def criteria(self) -> typing.Dict:
        return {
            self.name: {

            }
        }

    def forward(self, y: Data, t: Data=None) -> typing.Any:
        
        rendered = self.render()
        data = y.merge(t)
        return rendered.format(data=data.render())


class Style(Evaluator):

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

    def __init__(self, llm: PromptModel, evaluation: Evaluator):

        self.llm = llm
        self.evaluation = evaluation
    
    def forward(self, y: Data, t: Data=None) -> Assessment:
        
        instructions = self.evaluation(
            y, t
        )
        
        result = self.llm(Message('System', instructions))
        return Assessment(**json.loads(result))


class Evaluator(Struct):
    """Evaluate the output of an AI model
    Typically with some kind of language output
    """
    results: typing.List[typing.Dict[str, Assessment]]
