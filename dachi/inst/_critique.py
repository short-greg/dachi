import typing
import typing

import pydantic
from ..base import Renderable

# Hypothesis
# Prior
# Critique
# Evaluation
# Params
# Optim
# Learner


# def create_tuples(**kwargs: typing.List):
#     """
#     Create a list of dictionaries from keyword arguments.
#     Each dictionary in the list will have keys corresponding to the keyword arguments
#     and values corresponding to the values of those keyword arguments, grouped by their
#     position in the input.
#     Args:
#         **kwargs: Arbitrary keyword arguments where each key is a string and each value is an iterable.
#     Returns:
#         List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a combination
#         of the input keyword arguments' values.
#     """
#     keys = kwargs.keys()
#     values = zip(*kwargs.values())
#     return [dict(zip(keys, value)) for value in values]


class Hypothesis(pydantic.BaseModel):

    pass


class Criterion(pydantic.BaseModel, Renderable):
    """
    A criterion is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    name: str = pydantic.Field(
        description="The name of the criterion"
    )
    description: str = pydantic.Field(
        description="A description of the criterion"
    )
    type_: typing.Type = pydantic.Field(
        default=str,
        description="The type of the value for the criterion."
    )

    def render(self) -> str:
        """
        Render the criterion as a string.
        Returns:
            str: The rendered criterion
        """
        return str({
            "name": self.name,
            "description": self.description,
            "type": self.type_
        })


class CompoundCriterion(pydantic.BaseModel, Renderable):
    """
    A compound criterion is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    The value is a combination of multiple criteria.
    """
    criteria: typing.List[Criterion] = pydantic.Field(
        description="A list of criteria"
    )

    def render(self) -> str:
        return str({
            "criteria": [c.render() for c in self.criteria]
        })


class LikertItem(pydantic.BaseModel):
    """
    A Likert scale is a type of rating scale used to measure attitudes or opinions.
    It is often used in surveys and questionnaires to assess the level of agreement or disagreement with a statement.
    The scale typically ranges from 1 to 5 or 1 to 7, with 1 representing strong disagreement and the highest number representing strong agreement.
    """
    description: str = pydantic.Field(
        description="A description of the item"
    )
    val: int = pydantic.Field(
        description="The value of the item"
    )
    

class LikertScaleCriterion(Criterion):

    """
    A Likert scale is a type of rating scale used to measure attitudes or opinions.
    It is often used in surveys and questionnaires to assess the level of agreement or disagreement with a statement.
    The scale typically ranges from 1 to 5 or 1 to 7, with 1 representing strong disagreement and the highest number representing strong agreement.
    """
    scale: typing.List[LikertItem] = pydantic.Field(
        description="A list of Likert items"
    )

    def render(self) -> str:
        return str({
            "scale": [c.render() for c in self.scale]
        })


class Evaluation(pydantic.BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    val: typing.Dict[str, int | float | str] = pydantic.Field(
        description="The evaluation of each criterion with the key specified by the named. Each criterion must be evaluated."
    )

    def to_record(self) -> typing.Dict:
        """
        Convert the evaluation to a record.
        Returns:
            typing.Dict: A record
        """
        return self.val

    def render(self) -> str:
        """
        Render the evaluation as a string.
        Returns:
            str: The rendered evaluation
        """
        return str(self.val)


class EvaluationBatch(pydantic.BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    evaluations: typing.Dict[int, Evaluation] = pydantic.Field(
        description="The evaluation each with the same index as the item evaluated."
    )

    def to_records(self) -> typing.List[typing.Dict]:
        """
        Convert the evaluations to a list of records.
        Returns:
            typing.List[typing.Dict]: A list of records
        """
        return [
            self.evaluations[i].to_record() 
            for i in self.evaluations.keys()
        ]

    def render(self) -> str:
        return str({
            "evaluations": [e.render() for e in self.evaluations]
        })


