from __future__ import annotations

from pydantic import Field
import pydantic
from ._field import BoundInt, BoundFloat, TextField, BoolField, DictField, ListField
from ._base import ResponseSpec
import typing as t

class Reason(ResponseSpec):
    """Reasoning text field."""
    reason: TextField


class Brainstorming(ResponseSpec):
    """Brainstorming text field."""
    ideas: ListField


class BaseCriterion(ResponseSpec):
    pass


CRITERION = t.TypeVar("CRITERION", bound=BaseCriterion)


class PassFailCriterion(BaseCriterion):
    """Dichotomous judgment criterion (pass/fail)."""

    passed: BoolField = Field(
        default_factory=lambda: BoolField(description="Whether the criterion was passed")
    )
    passing_criteria: TextField | None = Field(
        default_factory=lambda: TextField(description="Description of the criteria for passing"
    ))
    @pydantic.field_validator('passing_criteria', mode='before')
    def validate_passing_criteria(cls, v):
        if isinstance(v, str):
            return TextField(description=v)
        return v


class LikertCriterion(BaseCriterion):
    """Likert scale criterion."""

    rating: BoundInt = Field(
        default_factory=lambda: BoundInt(min_val=1, max_val=5, description="Likert scale rating for the criterion")
    )


class NumericalRatingCriterion(BaseCriterion):
    """Numerical rating scale criterion with continuous values."""

    score: BoundFloat = Field(
        default_factory=lambda: BoundFloat(min_val=0.0, max_val=10.0, description="Numerical score for the criterion")
    )


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


class HolisticRubricCriterion(BaseCriterion):
    """Holistic rubric criterion with single overall level."""

    level: TextField = Field(
        default_factory=lambda: TextField(description="The overall level achieved")
    )
    level_index: BoundInt = Field(
        default_factory=lambda: BoundInt(min_val=1, max_val=5, description="Index of the overall level")
    )


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


class NarrativeCriterion(BaseCriterion):
    """Narrative/qualitative criterion."""

    narrative: TextField = Field(
        default_factory=lambda: TextField(description="The narrative evaluation text")
    )


class ComparativeCriterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    mode: TextField = Field(
        default_factory=lambda: TextField(description="The comparison mode: pairwise, ranking, or best_of")
    ) # "pairwise", "ranking", or "best_of"
    result: TextField = Field(
        default_factory=lambda: TextField(description="The comparison result")
    ) # Winner ID, ranking list (as string), or best ID


class CrispCiterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    result: BoolField = Field(
        default_factory=lambda: BoolField(description="The element either belongs to the set or does not")
    )


class FuzzyCiterion(BaseCriterion):
    """Comparative criterion for ranking/comparing outputs."""

    result: BoundFloat = Field(
        default_factory=lambda: BoundFloat(0.0, 1.0, description="The degree of membership of the element between 0 and 1")
    )
