import typing
import typing

import pydantic
from pydantic import ConfigDict, PrivateAttr, Field, create_model
from typing import Type, List

from ..core import Renderable

# Hypothesis
# Prior
# Critique
# Evaluation
# Params
# Optim
# Learner


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



class BaseCriterion(pydantic.BaseModel):
    """Base criterion that generates evaluation schemas in model_post_init.

    Subclasses must implement _create_evaluation_schema() to define the structure
    of evaluations for this criterion.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None

    _evaluation_schema: Type[pydantic.BaseModel] | None = PrivateAttr(default=None)
    _batch_evaluation_schema: Type[pydantic.BaseModel] | None = PrivateAttr(default=None)

    @property
    def evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Get the single evaluation schema."""
        if self._evaluation_schema is None:
            raise RuntimeError("evaluation_schema not initialized")
        return self._evaluation_schema

    @property
    def batch_evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Get the batch evaluation schema."""
        if self._batch_evaluation_schema is None:
            raise RuntimeError("batch_evaluation_schema not initialized")
        return self._batch_evaluation_schema

    def model_post_init(self, __context):
        """Generate evaluation schemas after initialization."""
        single_schema = self._create_evaluation_schema()
        object.__setattr__(self, '_evaluation_schema', single_schema)

        batch_schema = self._create_batch_evaluation_schema(single_schema)
        object.__setattr__(self, '_batch_evaluation_schema', batch_schema)

    def _create_evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Override in subclasses to create single evaluation schema."""
        raise NotImplementedError("Subclasses must implement _create_evaluation_schema")

    def _create_batch_evaluation_schema(self, single_schema: Type[pydantic.BaseModel]) -> Type[pydantic.BaseModel]:
        """Create batch schema (default: wraps single schema in list with criterion_name)."""
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            criterion_name=(str, Field(default=self.name)),
            evaluations=(List[single_schema], Field(
                description="List of evaluations, one per output"
            )),
            __base__=pydantic.BaseModel
        )

    def render(self) -> str:
        """Render criterion for prompt (override in subclasses)."""
        if self.description:
            return f"{self.name}: {self.description}"
        return self.name


class PassFailCriterion(BaseCriterion):
    """Dichotomous judgment criterion (meets standard or doesn't)."""

    passing_criteria: str | None = None

    def _create_evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Create PassFail evaluation schema."""
        return create_model(
            f'{self.name.replace(" ", "_")}PassFailEvaluation',
            criterion_name=(str, Field(default=self.name)),
            passed=(bool, Field(description="Whether output passes")),
            reason=(str, Field(description="Reason for pass or fail")),
            __base__=pydantic.BaseModel
        )

    def render(self) -> str:
        """Render criterion for prompt."""
        base = super().render()
        if self.passing_criteria:
            return f"{base}\nPassing criteria: {self.passing_criteria}"
        return base


class LikertCriterion(BaseCriterion):
    """Ordinal rating scale criterion for attitudes/opinions."""

    scale: List[LikertItem]

    def _create_evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Create Likert evaluation schema with rating constraints."""
        # Get min and max from scale
        scale_values = [item.val for item in self.scale]
        min_val = min(scale_values)
        max_val = max(scale_values)

        return create_model(
            f'{self.name.replace(" ", "_")}LikertEvaluation',
            criterion_name=(str, Field(default=self.name)),
            rating=(int, Field(description="Rating value", ge=min_val, le=max_val)),
            explanation=(str, Field(description="Explanation for rating")),
            __base__=pydantic.BaseModel
        )

    def render(self) -> str:
        """Render criterion for prompt including scale items."""
        base = super().render()
        scale_text = "\n".join(
            f"  {item.val}: {item.description}" for item in self.scale
        )
        return f"{base}\nScale:\n{scale_text}"


class NumericalRatingCriterion(BaseCriterion):
    """Numerical rating scale criterion with continuous values."""

    min_value: float
    max_value: float

    def _create_evaluation_schema(self) -> Type[pydantic.BaseModel]:
        """Create numerical rating evaluation schema."""
        return create_model(
            f'{self.name.replace(" ", "_")}NumericalEvaluation',
            criterion_name=(str, Field(default=self.name)),
            score=(float, Field(description="Numerical score", ge=self.min_value, le=self.max_value)),
            explanation=(str, Field(description="Explanation for score")),
            __base__=pydantic.BaseModel
        )

    def render(self) -> str:
        """Render criterion for prompt including rating range."""
        base = super().render()
        return f"{base}\nRating range: {self.min_value} to {self.max_value}"
