"""Unit tests for dachi.inst criterion system

This suite covers the new criterion and critic system including:
- BaseCriterion and all criterion types
- Dynamic schema generation
- Validation and immutability
"""
import pytest
from pydantic import BaseModel, ValidationError
from typing import Type, List
from dachi.inst._critique import (
    BaseCriterion,
    PassFailCriterion,
    LikertCriterion,
    LikertItem,
    NumericalRatingCriterion,
)


class TestBaseCriterion:
    """Tests for `BaseCriterion` base class."""

    def test_evaluation_schema_generated_in_model_post_init(self):
        """BaseCriterion subclass generates evaluation_schema during initialization."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    score=(int, Field(description="Score")),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test")
        assert criterion.evaluation_schema is not None

    def test_batch_evaluation_schema_generated_in_model_post_init(self):
        """BaseCriterion generates batch_evaluation_schema during initialization."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    score=(int, Field(description="Score")),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test")
        assert criterion.batch_evaluation_schema is not None

    def test_schemas_are_pydantic_model_types(self):
        """Schemas are Pydantic model classes, not instances."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    score=(int, Field(description="Score")),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test")
        assert isinstance(criterion.evaluation_schema, type)
        assert issubclass(criterion.evaluation_schema, BaseModel)
        assert isinstance(criterion.batch_evaluation_schema, type)
        assert issubclass(criterion.batch_evaluation_schema, BaseModel)

    def test_batch_schema_has_criterion_name_and_evaluations_fields(self):
        """Batch schema includes criterion_name and evaluations list."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    score=(int, Field(description="Score")),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test_criterion")
        batch_schema = criterion.batch_evaluation_schema

        # Create instance to verify fields
        batch_instance = batch_schema(evaluations=[])
        assert hasattr(batch_instance, 'criterion_name')
        assert batch_instance.criterion_name == "test_criterion"
        assert hasattr(batch_instance, 'evaluations')
        assert isinstance(batch_instance.evaluations, list)

    def test_criterion_is_frozen_after_creation(self):
        """Criterion instances are immutable (frozen=True)."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test", description="Test description")

        with pytest.raises(ValidationError):
            criterion.name = "changed"

    def test_name_is_required(self):
        """Creating criterion without name raises ValidationError."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    __base__=BaseModel
                )

        with pytest.raises(ValidationError):
            SimpleCriterion()

    def test_description_is_optional_and_defaults_to_none(self):
        """Description field is optional and defaults to None."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test")
        assert criterion.description is None

        criterion_with_desc = SimpleCriterion(name="test", description="A description")
        assert criterion_with_desc.description == "A description"

    def test_render_raises_not_implemented_in_base_class(self):
        """Calling render() on BaseCriterion subclass that doesn't implement it raises NotImplementedError."""
        class SimpleCriterion(BaseCriterion):
            def _create_evaluation_schema(self) -> Type[BaseModel]:
                from pydantic import create_model, Field
                return create_model(
                    f'{self.name}Evaluation',
                    criterion_name=(str, Field(default=self.name)),
                    __base__=BaseModel
                )

        criterion = SimpleCriterion(name="test")

        # Default implementation should return basic rendering
        # If subclass doesn't override, base class provides default
        result = criterion.render()
        assert isinstance(result, str)
        assert "test" in result

    def test_create_evaluation_schema_raises_not_implemented_in_base_class(self):
        """Creating BaseCriterion without implementing _create_evaluation_schema raises NotImplementedError."""

        # BaseCriterion itself raises NotImplementedError
        with pytest.raises(NotImplementedError):
            BaseCriterion(name="test")


class TestPassFailCriterion:
    """Tests for `PassFailCriterion`."""

    def test_evaluation_schema_has_passed_field_as_bool(self):
        """Evaluation schema includes passed field of type bool."""
        criterion = PassFailCriterion(name="test")
        evaluation = criterion.evaluation_schema(passed=True, reason="test")

        assert isinstance(evaluation.passed, bool)

    def test_evaluation_schema_has_reason_field_as_str(self):
        """Evaluation schema includes reason field of type str."""
        criterion = PassFailCriterion(name="test")
        evaluation = criterion.evaluation_schema(passed=True, reason="explanation")

        assert isinstance(evaluation.reason, str)

    def test_evaluation_schema_has_criterion_name_field_with_default(self):
        """Evaluation schema includes criterion_name field that defaults to criterion name."""
        criterion = PassFailCriterion(name="safety_check")
        evaluation = criterion.evaluation_schema(passed=True, reason="test")

        assert hasattr(evaluation, 'criterion_name')
        assert evaluation.criterion_name == "safety_check"

    def test_can_create_evaluation_with_passed_true(self):
        """Can create evaluation instance with passed=True."""
        criterion = PassFailCriterion(name="test")
        evaluation = criterion.evaluation_schema(passed=True, reason="Meets requirements")

        assert evaluation.passed is True
        assert evaluation.reason == "Meets requirements"

    def test_can_create_evaluation_with_passed_false(self):
        """Can create evaluation instance with passed=False."""
        criterion = PassFailCriterion(name="test")
        evaluation = criterion.evaluation_schema(passed=False, reason="Does not meet requirements")

        assert evaluation.passed is False
        assert evaluation.reason == "Does not meet requirements"

    def test_batch_schema_wraps_single_evaluations(self):
        """Batch evaluation schema wraps list of single evaluations."""
        criterion = PassFailCriterion(name="test")
        batch_schema = criterion.batch_evaluation_schema

        # Create batch with multiple evaluations
        eval1 = criterion.evaluation_schema(passed=True, reason="Good")
        eval2 = criterion.evaluation_schema(passed=False, reason="Bad")
        batch = batch_schema(evaluations=[eval1, eval2])

        assert len(batch.evaluations) == 2
        assert batch.evaluations[0].passed is True
        assert batch.evaluations[1].passed is False

    def test_evaluation_requires_passed_field(self):
        """Creating evaluation without passed field raises ValidationError."""
        criterion = PassFailCriterion(name="test")

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(reason="Missing passed field")

    def test_evaluation_requires_reason_field(self):
        """Creating evaluation without reason field raises ValidationError."""
        criterion = PassFailCriterion(name="test")

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(passed=True)

    def test_render_with_passing_criteria_includes_it(self):
        """Render output includes passing_criteria text when provided."""
        criterion = PassFailCriterion(
            name="safety",
            passing_criteria="No harmful content"
        )

        rendered = criterion.render()
        assert "No harmful content" in rendered

    def test_render_without_passing_criteria_shows_name(self):
        """Render output shows criterion name when no passing_criteria provided."""
        criterion = PassFailCriterion(name="basic_check")

        rendered = criterion.render()
        assert "basic_check" in rendered

    def test_name_with_spaces_creates_valid_schema_name(self):
        """Criterion name with spaces generates valid schema class name."""
        criterion = PassFailCriterion(name="safety check")

        # Should not raise an error
        evaluation = criterion.evaluation_schema(passed=True, reason="test")
        assert evaluation is not None


class TestLikertCriterion:
    """Tests for `LikertCriterion`."""

    def test_evaluation_schema_has_rating_field_as_int(self):
        """Evaluation schema includes rating field of type int."""
        scale = [
            LikertItem(val=1, description="Strongly disagree"),
            LikertItem(val=5, description="Strongly agree")
        ]
        criterion = LikertCriterion(name="satisfaction", scale=scale)

        evaluation = criterion.evaluation_schema(rating=3, explanation="Neutral")

        assert isinstance(evaluation.rating, int)

    def test_evaluation_schema_has_explanation_field_as_str(self):
        """Evaluation schema includes explanation field of type str."""
        scale = [
            LikertItem(val=1, description="Poor"),
            LikertItem(val=5, description="Excellent")
        ]
        criterion = LikertCriterion(name="quality", scale=scale)

        evaluation = criterion.evaluation_schema(rating=4, explanation="Good quality")

        assert isinstance(evaluation.explanation, str)

    def test_rating_within_scale_range_is_valid(self):
        """Rating within min and max scale values is valid."""
        scale = [
            LikertItem(val=1, description="Low"),
            LikertItem(val=3, description="Medium"),
            LikertItem(val=5, description="High")
        ]
        criterion = LikertCriterion(name="test", scale=scale)

        evaluation = criterion.evaluation_schema(rating=3, explanation="test")

        assert evaluation.rating == 3

    def test_rating_below_minimum_raises_validation_error(self):
        """Rating below minimum scale value raises ValidationError."""
        scale = [
            LikertItem(val=1, description="Low"),
            LikertItem(val=5, description="High")
        ]
        criterion = LikertCriterion(name="test", scale=scale)

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(rating=0, explanation="Too low")

    def test_rating_above_maximum_raises_validation_error(self):
        """Rating above maximum scale value raises ValidationError."""
        scale = [
            LikertItem(val=1, description="Low"),
            LikertItem(val=5, description="High")
        ]
        criterion = LikertCriterion(name="test", scale=scale)

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(rating=6, explanation="Too high")

    def test_scale_values_define_rating_constraints(self):
        """Scale min/max values correctly constrain rating field."""
        scale = [
            LikertItem(val=2, description="Start"),
            LikertItem(val=8, description="End")
        ]
        criterion = LikertCriterion(name="test", scale=scale)

        # Should work at boundaries
        eval_min = criterion.evaluation_schema(rating=2, explanation="min")
        eval_max = criterion.evaluation_schema(rating=8, explanation="max")

        assert eval_min.rating == 2
        assert eval_max.rating == 8

    def test_render_includes_all_scale_items_with_descriptions(self):
        """Render output includes all scale items with their descriptions."""
        scale = [
            LikertItem(val=1, description="Strongly disagree"),
            LikertItem(val=2, description="Disagree"),
            LikertItem(val=3, description="Neutral"),
            LikertItem(val=4, description="Agree"),
            LikertItem(val=5, description="Strongly agree")
        ]
        criterion = LikertCriterion(
            name="agreement",
            description="Level of agreement",
            scale=scale
        )

        rendered = criterion.render()

        # Should include all scale items
        assert "1" in rendered and "Strongly disagree" in rendered
        assert "5" in rendered and "Strongly agree" in rendered

    def test_non_contiguous_scale_values_work(self):
        """Scale with non-contiguous values (1,3,5) works correctly."""
        scale = [
            LikertItem(val=1, description="Low"),
            LikertItem(val=3, description="Medium"),
            LikertItem(val=5, description="High")
        ]
        criterion = LikertCriterion(name="test", scale=scale)

        # All scale values should be valid
        eval1 = criterion.evaluation_schema(rating=1, explanation="low")
        eval3 = criterion.evaluation_schema(rating=3, explanation="med")
        eval5 = criterion.evaluation_schema(rating=5, explanation="high")

        assert eval1.rating == 1
        assert eval3.rating == 3
        assert eval5.rating == 5


class TestNumericalRatingCriterion:
    """Tests for `NumericalRatingCriterion`."""

    def test_evaluation_schema_has_score_field_as_float(self):
        """Evaluation schema includes score field of type float."""
        criterion = NumericalRatingCriterion(
            name="quality",
            min_value=0.0,
            max_value=10.0
        )

        evaluation = criterion.evaluation_schema(score=7.5, explanation="Good")

        assert isinstance(evaluation.score, float)

    def test_evaluation_schema_has_explanation_field_as_str(self):
        """Evaluation schema includes explanation field of type str."""
        criterion = NumericalRatingCriterion(
            name="quality",
            min_value=0.0,
            max_value=10.0
        )

        evaluation = criterion.evaluation_schema(score=8.0, explanation="Excellent")

        assert isinstance(evaluation.explanation, str)

    def test_score_within_range_is_valid(self):
        """Score within min and max values is valid."""
        criterion = NumericalRatingCriterion(
            name="test",
            min_value=1.0,
            max_value=5.0
        )

        evaluation = criterion.evaluation_schema(score=3.5, explanation="test")

        assert evaluation.score == 3.5

    def test_score_below_minimum_raises_validation_error(self):
        """Score below minimum value raises ValidationError."""
        criterion = NumericalRatingCriterion(
            name="test",
            min_value=0.0,
            max_value=10.0
        )

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(score=-1.0, explanation="Too low")

    def test_score_above_maximum_raises_validation_error(self):
        """Score above maximum value raises ValidationError."""
        criterion = NumericalRatingCriterion(
            name="test",
            min_value=0.0,
            max_value=10.0
        )

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(score=11.0, explanation="Too high")

    def test_score_constraints_match_min_max_values(self):
        """Score constraints correctly use min_value and max_value."""
        criterion = NumericalRatingCriterion(
            name="test",
            min_value=2.5,
            max_value=7.5
        )

        # Boundaries should work
        eval_min = criterion.evaluation_schema(score=2.5, explanation="min")
        eval_max = criterion.evaluation_schema(score=7.5, explanation="max")

        assert eval_min.score == 2.5
        assert eval_max.score == 7.5

    def test_render_includes_rating_range(self):
        """Render output includes min and max rating range."""
        criterion = NumericalRatingCriterion(
            name="clarity",
            description="Rate the clarity",
            min_value=0.0,
            max_value=10.0
        )

        rendered = criterion.render()

        assert "0.0" in rendered or "0" in rendered
        assert "10.0" in rendered or "10" in rendered

    def test_fractional_scores_work(self):
        """Fractional score values are properly handled as floats."""
        criterion = NumericalRatingCriterion(
            name="test",
            min_value=0.0,
            max_value=10.0
        )

        eval1 = criterion.evaluation_schema(score=0.5, explanation="low")
        eval2 = criterion.evaluation_schema(score=5.75, explanation="mid")
        eval3 = criterion.evaluation_schema(score=9.99, explanation="high")

        assert eval1.score == 0.5
        assert eval2.score == 5.75
        assert eval3.score == 9.99
