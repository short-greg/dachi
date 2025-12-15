"""Unit tests for new criterion system (_criterion.py)

This suite covers the EvalField-based criterion architecture including:
- BaseCriterion auto-generation
- Concrete criterion types (PassFail, Likert, NumericalRating)
- Critic evaluation executor
"""
import pytest
from pydantic import BaseModel, ValidationError
from dachi.inst._criterion import (
    PassFailCriterion,
    LikertCriterion,
    NumericalRatingCriterion,
    BoundInt,
    BoundFloat,
    TextField,
    BoolField,
)


class TestBaseCriterion:
    """Tests for BaseCriterion auto-generation."""

    def test_evaluation_schema_is_generated_automatically(self):
        """BaseCriterion generates evaluation_schema in model_post_init."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        assert criterion.response_schema is not None
        assert isinstance(criterion.response_schema, type)
        assert issubclass(criterion.response_schema, BaseModel)

    def test_batch_evaluation_schema_wraps_single_schema_in_list(self):
        """Batch schema contains responses field as List[SingleSchema]."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        batch_schema = criterion.batch_response_schema
        batch_instance = batch_schema(responses=[])

        assert hasattr(batch_instance, 'responses')
        assert isinstance(batch_instance.responses, list)

    def test_criterion_is_immutable_after_creation(self):
        """Criterion with frozen=True raises error when modified."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        with pytest.raises(ValidationError):
            criterion.name = "changed"

    def test_evaluation_schema_includes_all_evalfield_fields(self):
        """Generated schema includes all EvalField-annotated fields."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5)
        )

        evaluation = criterion.response_schema(rating=3)

        assert hasattr(evaluation, 'rating')
        assert evaluation.rating == 3

    def test_schema_name_sanitizes_criterion_name_with_spaces(self):
        """Criterion name with spaces generates valid schema class name."""

        criterion = PassFailCriterion(
            name="safety check",
            passed=BoolField()
        )

        evaluation = criterion.response_schema(passed=True, passing_criteria="All good")
        assert evaluation is not None


class TestLikertCriterion:
    """Tests for LikertCriterion."""

    def test_rating_field_enforces_min_and_max_bounds(self):
        """Rating below min or above max raises ValidationError."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5)
        )

        with pytest.raises(ValidationError):
            criterion.response_schema(rating=0)
        with pytest.raises(ValidationError):
            criterion.response_schema(rating=6)

    def test_rating_at_boundaries_is_valid(self):
        """Rating equal to min_val or max_val is valid."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5)
        )

        eval_min = criterion.response_schema(rating=1)
        eval_max = criterion.response_schema(rating=5)
        assert eval_min.rating == 1
        assert eval_max.rating == 5

    # def test_render_includes_scale_information(self):
    #     """Render output includes min and max rating values."""
    #     criterion = LikertCriterion(
    #         name="quality",
    #         description="Rate quality",
    #         rating=BoundInt(min_val=1, max_val=5))

    #     rendered = criterion.render()

    #     assert "quality" in rendered
    #     assert "1" in rendered
    #     assert "5" in rendered


class TestPassFailCriterion:
    """Tests for PassFailCriterion."""

    def test_evaluation_schema_has_passed_fields(self):
        """Evaluation schema includes passed (bool)."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            passing_criteria=TextField(
                description="Criteria for passing the test"
            )
        )

        evaluation = criterion.response_schema(passed=True, passing_criteria="All checks passed")

        assert isinstance(evaluation.passed, bool)
        assert evaluation.passed is True

    # def test_render_includes_passing_criteria_when_provided(self):
    #     """Render output includes passing_criteria if present."""
    #     criterion = PassFailCriterion(
    #         name="safety",
    #         passing_criteria="No harmful content",
    #         passed=BoolField()
    #     )

    #     rendered = criterion.render()

    #     assert "safety" in rendered
    #     assert "No harmful content" in rendered


class TestNumericalRatingCriterion:
    """Tests for NumericalRatingCriterion."""

    def test_score_field_enforces_float_bounds(self):
        """Score below min or above max raises ValidationError."""
        criterion = NumericalRatingCriterion(
            name="test",
            score=BoundFloat(min_val=0.0, max_val=10.0)
        )

        with pytest.raises(ValidationError):
            criterion.response_schema(score=-0.1)

        with pytest.raises(ValidationError):
            criterion.response_schema(score=10.1)

    def test_fractional_scores_work_correctly(self):
        """Fractional score values are properly validated as floats."""
        criterion = NumericalRatingCriterion(
            name="test",
            score=BoundFloat(min_val=0.0, max_val=10.0)
        )

        evaluation = criterion.response_schema(score=7.5)

        assert evaluation.score == 7.5
        assert isinstance(evaluation.score, float)
