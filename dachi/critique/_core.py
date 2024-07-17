from abc import ABC, abstractmethod
from .. import converse
import typing
from .._core import Struct


class Assessment(Struct):
    """Evaluate the output of an AI model
    Typically with some kind of language output
    """

    name: str
    evaluation: typing.Any
    info: typing.Dict[str, typing.Any]


class JointAssessment(Assessment):
    """Store multiple assessments by an AI model 

    """

    assessments: typing.Dict[str, Assessment]

    def __getitem__(self, key: str) -> Assessment:
        """Get the assessement associated with a string

        Args:
            key (str): the name of the assessment

        Returns:
            Assessment: the assessment for that name
        """
        
        return self.assessments[key]
    
    def __setitem__(self, key: str, value: Assessment) -> Assessment:
        """Set the 

        Args:
            key (str): _description_
            value (Assessment): _description_

        Returns:
            Assessment: The value passed in
        """
        self.assessments[key] = value
        return value


class AssessmentList(object):

    def __init__(self, assessments: Assessment):

        self.assessments = assessments

    def to_records(self) -> typing.List[typing.Dict]:

        pass


class Assessor(converse.Module, ABC):

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
