from __future__ import annotations

from pydantic import Field
import pydantic
from ._field import BoundInt, BoundFloat, TextField, BoolField, DictField, ListField
from ._base import BaseCriterion, CriterionMixin


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
