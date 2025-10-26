from __future__ import annotations

import typing
from abc import abstractmethod
from typing import Type, List
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model
from ..core import Renderable


class EvalField(BaseModel):
    """Base class for evaluation field descriptors."""

    description: str | None = None

    @abstractmethod
    def get_field(self) -> tuple:
        """Return (type, Field(...)) tuple for create_model.

        Returns:
            tuple: (field_type, Field(...)) for use in create_model
        """
        pass


class BoundInt(EvalField):
    """Integer field with min/max bounds."""

    min_val: int
    max_val: int

    def get_field(self) -> tuple:
        return (int, Field(description=self.description, ge=self.min_val, le=self.max_val))


class BoundFloat(EvalField):
    """Float field with min/max bounds."""

    min_val: float
    max_val: float

    def get_field(self) -> tuple:
        return (float, Field(description=self.description, ge=self.min_val, le=self.max_val))


class TextField(EvalField):
    """String text field."""

    def get_field(self) -> tuple:
        return (str, Field(description=self.description))


class BoolField(EvalField):
    """Boolean field."""

    def get_field(self) -> tuple:
        return (bool, Field(description=self.description))


class DictField(EvalField):
    """Dictionary field for dynamic key-value pairs."""

    value_type: Type = str

    def get_field(self) -> tuple:
        return (typing.Dict[str, self.value_type], Field(description=self.description))


class ListField(EvalField):
    """List field."""

    item_type: Type = str

    def get_field(self) -> tuple:
        return (typing.List[self.item_type], Field(description=self.description, default_factory=list))


class BaseCriterion(BaseModel, Renderable):
    """Base class for all criteria. Auto-generates evaluation schemas from EvalFields."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None

    _evaluation_schema: Type[BaseModel] | None = PrivateAttr(default=None)
    _batch_evaluation_schema: Type[BaseModel] | None = PrivateAttr(default=None)

    @property
    def evaluation_schema(self) -> Type[BaseModel]:
        """Get the single evaluation schema."""
        if self._evaluation_schema is None:
            raise RuntimeError("evaluation_schema not initialized")
        return self._evaluation_schema

    @property
    def batch_evaluation_schema(self) -> Type[BaseModel]:
        """Get the batch evaluation schema."""
        if self._batch_evaluation_schema is None:
            raise RuntimeError("batch_evaluation_schema not initialized")
        return self._batch_evaluation_schema

    def model_post_init(self, __context) -> None:
        """Auto-generate evaluation schemas from EvalFields."""
        single = self._create_single()
        batch = self._create_batch(single)

        object.__setattr__(self, '_evaluation_schema', single)
        object.__setattr__(self, '_batch_evaluation_schema', batch)

    def _create_single(self) -> Type[BaseModel]:
        """Create single evaluation schema by introspecting EvalFields."""
        fields = {'criterion_name': (str, Field(default=self.name))}

        # Use model_fields to find EvalField annotations
        for field_name, field_info in self.model_fields.items():
            field_value = getattr(self, field_name)
            if isinstance(field_value, EvalField):
                fields[field_name] = field_value.get_field()

        # include eval_type in the model that is 
        # "single" or "batch"
        return create_model(
            f'{self.name.replace(" ", "_")}Evaluation',
            **fields,
            __base__=Evaluation,
        )

    def _create_batch(self, single_schema: Type[BaseModel]) -> Type[BaseModel]:
        """Create batch evaluation schema."""
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            criterion_name=(str, Field(default=self.name)),
            evaluations=(List[single_schema], Field(description="List of evaluations")),
            __base__=BatchEvaluation
        )

    @abstractmethod
    def render(self) -> str:
        """Render criterion for prompt."""
        pass


class PassFailCriterion(BaseCriterion):
    """Dichotomous judgment criterion (pass/fail)."""

    passed: BoolField
    reason: TextField
    passing_criteria: str | None = None

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        if self.passing_criteria:
            return f"{base}\nPassing criteria: {self.passing_criteria}"
        return base


class LikertCriterion(BaseCriterion):
    """Likert scale criterion."""

    rating: BoundInt
    explanation: TextField

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nScale: {self.rating.min_val} to {self.rating.max_val}"


class NumericalRatingCriterion(BaseCriterion):
    """Numerical rating scale criterion with continuous values."""

    score: BoundFloat
    explanation: TextField

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nRating range: {self.score.min_val} to {self.score.max_val}"


class ChecklistCriterion(BaseCriterion):
    """Checklist criterion with multiple boolean checks."""

    items: DictField  # Dict[str, bool]
    missing_items: ListField  # List[str]

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class HolisticRubricCriterion(BaseCriterion):
    """Holistic rubric criterion with single overall level."""

    level: TextField
    level_index: BoundInt
    explanation: TextField

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nLevels: {self.level_index.min_val} to {self.level_index.max_val}"


class AnalyticRubricCriterion(BaseCriterion):
    """Analytic rubric criterion with multiple dimensions."""

    dimensions: DictField  # Dict[str, dict] with score and explanation per dimension
    overall_score: BoundFloat

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class NarrativeCriterion(BaseCriterion):
    """Narrative/qualitative criterion."""

    narrative: TextField

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class ComparativeCriterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    mode: TextField  # "pairwise", "ranking", or "best_of"
    result: TextField  # Winner ID, ranking list (as string), or best ID
    explanation: TextField

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class Evaluation(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """

    def to_record(self) -> typing.Dict:
        """
        Convert the evaluation to a record.
        Returns:
            typing.Dict: A record
        """
        return self.model_dump()

    def render(self) -> str:
        """
        Render the evaluation as a string.
        Returns:
            str: The rendered evaluation
        """
        return str(self.model_dump())


class BatchEvaluation(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    evaluations: typing.List[Evaluation]

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



