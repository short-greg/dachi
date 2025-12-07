from __future__ import annotations

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model
import pydantic
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

    value_type: t.Type = str

    def get_field(self) -> tuple:
        return (t.Dict[str, self.value_type], Field(description=self.description))


class ListField(EvalField):
    """List field."""

    item_type: t.Type = str

    def get_field(self) -> tuple:
        return (t.List[self.item_type], Field(description=self.description, default_factory=list))


class BaseCriterion(BaseModel, Renderable):
    """Base class for all criteria. Auto-generates evaluation schemas from EvalFields."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None

    _evaluation_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)
    _batch_evaluation_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)

    @property
    def evaluation_schema(self) -> t.Type[BaseModel]:
        """Get the single evaluation schema."""
        if self._evaluation_schema is None:
            raise RuntimeError("evaluation_schema not initialized")
        return self._evaluation_schema

    @property
    def batch_evaluation_schema(self) -> t.Type[BaseModel]:
        """Get the batch evaluation schema."""
        if self._batch_evaluation_schema is None:
            raise RuntimeError("batch_evaluation_schema not initialized")
        return self._batch_evaluation_schema

    def model_post_init(self, __context) -> None:
        """Auto-generate evaluation schemas from EvalFields."""
        super().model_post_init(__context)
        single = self._create_single()
        batch = self._create_batch(single)

        object.__setattr__(self, '_evaluation_schema', single)
        object.__setattr__(self, '_batch_evaluation_schema', batch)

    def _create_single(self) -> t.Type[BaseModel]:
        """Create single evaluation schema by introspecting EvalFields.

        Note: criterion_name uses Optional[str] with default to support OpenAI strict mode.
        OpenAI strict mode requires all properties in 'required' array but doesn't support
        traditional defaults. Instead, we use nullable types (str | None) and apply the
        default after validation if the LLM returns null.
        """
        fields = {
            'criterion_name': (
                t.Optional[str],
                Field(default=self.name, description="Name of the criterion")
            )
        }

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

    def _create_batch(self, single_schema: t.Type[BaseModel]) -> t.Type[BaseModel]:
        """Create batch evaluation schema.

        Note: criterion_name uses Optional[str] with default to support OpenAI strict mode.
        """
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            criterion_name=(t.Optional[str], Field(default=self.name, description="Name of the criterion")),
            evaluations=(t.List[single_schema], Field(description="List of evaluations")),
            __base__=BatchEvaluation
        )

    @abstractmethod
    def render(self) -> str:
        """Render criterion for prompt."""
        pass


CRITERION = t.TypeVar("CRITERION", bound=BaseCriterion)


class CriterionMixin(BaseModel):
    """Mixin for criteria with reasoning."""
    pass


class Reason(CriterionMixin):
    """Reasoning text field."""
    reason: TextField


class Brainstorming(CriterionMixin):
    """Brainstorming text field."""
    ideas: ListField


class PassFailCriterion(BaseCriterion):
    """Dichotomous judgment criterion (pass/fail)."""

    passed: BoolField = Field(
        default_factory=lambda: BoolField(description="Whether the criterion was passed")
    )
    passing_criteria: TextField | None

    @pydantic.field_validator('passing_criteria', mode='before')
    def validate_passing_criteria(cls, v):
        if isinstance(v, str):
            return TextField(description=v)
        return v

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        if self.passing_criteria:
            return f"{base}\nPassing criteria: {self.passing_criteria}"
        return base


class LikertCriterion(BaseCriterion):
    """Likert scale criterion."""

    rating: BoundInt = Field(
        default_factory=lambda: BoundInt(min_val=1, max_val=5, description="Likert scale rating for the criterion")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nScale: {self.rating.min_val} to {self.rating.max_val}"


class NumericalRatingCriterion(BaseCriterion):
    """Numerical rating scale criterion with continuous values."""

    score: BoundFloat = Field(
        default_factory=lambda: BoundFloat(min_val=0.0, max_val=10.0, description="Numerical score for the criterion")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nRating range: {self.score.min_val} to {self.score.max_val}"


class ChecklistCriterion(BaseCriterion):
    """Checklist criterion with multiple boolean checks."""

    items: DictField = Field(
        default_factory=lambda: DictField(
            value_type=bool,
            description="Checklist items with boolean pass/fail values"
        )
    ) # Dict[str, bool]
    missing_items: ListField = Field(
        default_factory=lambda: ListField(item_type=str, description="List of missing checklist items")
    )  # List[str]

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class HolisticRubricCriterion(BaseCriterion):
    """Holistic rubric criterion with single overall level."""

    level: TextField = Field(
        default_factory=lambda: TextField(description="The overall level achieved")
    )
    level_index: BoundInt = Field(
        default_factory=lambda: BoundInt(min_val=1, max_val=5, description="Index of the overall level")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return f"{base}\nLevels: {self.level_index.min_val} to {self.level_index.max_val}"


class AnalyticRubricCriterion(BaseCriterion):
    """Analytic rubric criterion with multiple dimensions."""

    dimensions: DictField = Field(
        default_factory=lambda: DictField(
            value_type=dict,
            description="Dimensions with scores and explanations"
        )
    ) # Dict[str, dict] with score and explanation per dimension
    overall_score: BoundFloat = Field(
        default_factory=lambda: BoundFloat(0.0, 10.0, description="Overall score across all dimensions")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class NarrativeCriterion(BaseCriterion):
    """Narrative/qualitative criterion."""

    narrative: TextField = Field(
        default_factory=lambda: TextField(description="The narrative evaluation text")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class ComparativeCriterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    mode: TextField = Field(
        default_factory=lambda: TextField(description="The comparison mode: pairwise, ranking, or best_of")
    ) # "pairwise", "ranking", or "best_of"
    result: TextField = Field(
        default_factory=lambda: TextField(description="The comparison result")
    ) # Winner ID, ranking list (as string), or best ID

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class CrispCiterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    result: BoolField = Field(
        default_factory=lambda: BoolField(description="The element either belongs to the set or does not")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class FuzzyCiterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    result: BoundFloat = Field(
        default_factory=lambda: BoundFloat(0.0, 1.0, description="The degree of membership of the element between 0 and 1")
    )

    def render(self) -> str:
        base = f"{self.name}: {self.description}" if self.description else self.name
        return base


class Evaluation(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    model_config = ConfigDict(extra='forbid')

    def to_record(self) -> t.Dict:
        """
        Convert the evaluation to a record.
        Returns:
            t.Dict: A record
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
    model_config = ConfigDict(extra='forbid')
    evaluations: t.List[Evaluation]

    def to_records(self) -> t.List[t.Dict]:
        """
        Convert the evaluations to a list of records.
        Returns:
            t.List[t.Dict]: A list of records
        """
        return [
            self.evaluations[i].to_record()
            for i in self.evaluations.keys()
        ]

    def render(self) -> str:
        return str({
            "evaluations": [e.render() for e in self.evaluations]
        })
