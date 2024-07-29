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


# class JointAssessment(Assessment):
#     """Store multiple assessments by an AI model 

#     """

#     assessments: typing.Dict[str, Assessment]

#     def __getitem__(self, key: str) -> Assessment:
#         """Get the assessement associated with a string

#         Args:
#             key (str): the name of the assessment

#         Returns:
#             Assessment: the assessment for that name
#         """
#         return self.assessments[key]
    
#     def __setitem__(self, key: str, value: Assessment) -> Assessment:
#         """Set 

#         Args:
#             key (str): _description_
#             value (Assessment): _description_

#         Returns:
#             Assessment: The value passed in
#         """
#         self.assessments[key] = value
#         return value


# class AssessmentList(object):

#     def __init__(self, assessments: Assessment):

#         self.assessments = assessments

#     def to_records(self) -> typing.List[typing.Dict]:

#         pass


# class Assessor(converse.Module, ABC):

#     @abstractmethod
#     def forward(self, *args, **kwargs) -> JointAssessment:
#         pass



# assessment['...']

# what is the important thing?
# that the language model understands the assessment
# that a human can understand the assessment.. for instance
#   it can rendered

# I think this is 

# {}
#




# class ConstraintEvaluation(Evaluation):

#     def out_format(self) -> typing.Dict:
        
#         return {
#             self.name: f'Evaluate the input according to '
#                        f'this constraint. {self.constraint}'
#         }

#     def forward(self, y: Data, t: Data=None) -> typing.Any:
        
#         rendered = self.render()
#         return rendered.format(data=y.render())


# class RegularizationEvaluation(Evaluation):

#     def out_format(self) -> typing.Dict:
        
#         return {
#             self.name: f'Evaluate each input according to '
#                        f'this audauregularization. {self.regularization}'
#         }
    
#     def forward(self, y: Data, t: Data=None) -> typing.Any:
        
#         rendered = self.render()
#         return rendered.format(data=y.render())