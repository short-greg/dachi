from abc import ABC, abstractmethod
from .. import process
import typing
from .._core import Struct


class Assessment(Struct):

    name: str
    evaluation: typing.Any
    info: typing.Dict[str, typing.Any]


class JointAssessment(Assessment):

    evaluation: typing.Dict[str, Assessment]

    def __getitem__(self, key: str):
        
        return self.evaluation[key]
    
    def __setitem__(self, key: str, value: Assessment) -> typing.Any:

        self.evaluation[key] = value
        return value


class AssessmentList(object):

    def __init__(self, assessments: Assessment):

        self.assessments = assessments

    def to_records(self) -> typing.List[typing.Dict]:

        pass


class Assessor(process.Module, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> JointAssessment:
        pass



# assessment['...']

# what is the important thing?
# that the language model understands the assessment
# that a human can understand the assessment.. for instance
#   it can rendered

# I think this is 

# {}
#
