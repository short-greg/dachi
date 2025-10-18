"""Unit tests for new criterion system (_criterion.py)

This suite covers the EvalField-based criterion architecture including:
- BaseCriterion auto-generation
- Concrete criterion types (PassFail, Likert, NumericalRating)
- Critic evaluation executor
"""
import pytest
from pydantic import BaseModel, ValidationError
from typing import Type
from dachi.inst._criterion import (
    BaseCriterion,
    PassFailCriterion,
    LikertCriterion,
    NumericalRatingCriterion,
    BoundInt,
    BoundFloat,
    TextField,
    BoolField,
    Critic,
)
from dachi.core import Prompt
from dachi.proc import Process


class MockResp:
    """Mock response from LLM."""
    def __init__(self, text: str):
        self.text = text


class MockProcess(Process):
    """Mock process for testing Critic."""

    def __init__(self, response_json: str):
        self.response_json = response_json
        self.last_prompt = None

    def forward(self, prompt, **kwargs):
        self.last_prompt = prompt
        return MockResp(self.response_json)


class TestBaseCriterion:
    """Tests for BaseCriterion auto-generation."""

    def test_evaluation_schema_is_generated_automatically(self):
        """BaseCriterion generates evaluation_schema in model_post_init."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        assert criterion.evaluation_schema is not None
        assert isinstance(criterion.evaluation_schema, type)
        assert issubclass(criterion.evaluation_schema, BaseModel)

    def test_batch_evaluation_schema_wraps_single_schema_in_list(self):
        """Batch schema contains evaluations field as List[SingleSchema]."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        batch_schema = criterion.batch_evaluation_schema
        batch_instance = batch_schema(evaluations=[])

        assert hasattr(batch_instance, 'evaluations')
        assert isinstance(batch_instance.evaluations, list)

    def test_criterion_is_immutable_after_creation(self):
        """Criterion with frozen=True raises error when modified."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        with pytest.raises(ValidationError):
            criterion.name = "changed"

    def test_evaluation_schema_includes_all_evalfield_fields(self):
        """Generated schema includes all EvalField-annotated fields."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5),
            explanation=TextField()
        )

        evaluation = criterion.evaluation_schema(rating=3, explanation="test")

        assert hasattr(evaluation, 'rating')
        assert hasattr(evaluation, 'explanation')
        assert evaluation.rating == 3
        assert evaluation.explanation == "test"

    def test_schema_name_sanitizes_criterion_name_with_spaces(self):
        """Criterion name with spaces generates valid schema class name."""
        criterion = PassFailCriterion(
            name="safety check",
            passed=BoolField(),
            reason=TextField()
        )

        evaluation = criterion.evaluation_schema(passed=True, reason="test")
        assert evaluation is not None


class TestLikertCriterion:
    """Tests for LikertCriterion."""

    def test_rating_field_enforces_min_and_max_bounds(self):
        """Rating below min or above max raises ValidationError."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5),
            explanation=TextField()
        )

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(rating=0, explanation="too low")

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(rating=6, explanation="too high")

    def test_rating_at_boundaries_is_valid(self):
        """Rating equal to min_val or max_val is valid."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5),
            explanation=TextField()
        )

        eval_min = criterion.evaluation_schema(rating=1, explanation="min")
        eval_max = criterion.evaluation_schema(rating=5, explanation="max")

        assert eval_min.rating == 1
        assert eval_max.rating == 5

    def test_explanation_field_is_required_string(self):
        """Creating evaluation without explanation raises ValidationError."""
        criterion = LikertCriterion(
            name="test",
            rating=BoundInt(min_val=1, max_val=5),
            explanation=TextField()
        )

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(rating=3)

    def test_render_includes_scale_information(self):
        """Render output includes min and max rating values."""
        criterion = LikertCriterion(
            name="quality",
            description="Rate quality",
            rating=BoundInt(min_val=1, max_val=5),
            explanation=TextField()
        )

        rendered = criterion.render()

        assert "quality" in rendered
        assert "1" in rendered
        assert "5" in rendered


class TestPassFailCriterion:
    """Tests for PassFailCriterion."""

    def test_evaluation_schema_has_passed_and_reason_fields(self):
        """Evaluation schema includes passed (bool) and reason (str)."""
        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        evaluation = criterion.evaluation_schema(passed=True, reason="good")

        assert isinstance(evaluation.passed, bool)
        assert isinstance(evaluation.reason, str)
        assert evaluation.passed is True
        assert evaluation.reason == "good"

    def test_render_includes_passing_criteria_when_provided(self):
        """Render output includes passing_criteria if present."""
        criterion = PassFailCriterion(
            name="safety",
            passing_criteria="No harmful content",
            passed=BoolField(),
            reason=TextField()
        )

        rendered = criterion.render()

        assert "safety" in rendered
        assert "No harmful content" in rendered


class TestNumericalRatingCriterion:
    """Tests for NumericalRatingCriterion."""

    def test_score_field_enforces_float_bounds(self):
        """Score below min or above max raises ValidationError."""
        criterion = NumericalRatingCriterion(
            name="test",
            score=BoundFloat(min_val=0.0, max_val=10.0),
            explanation=TextField()
        )

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(score=-0.1, explanation="too low")

        with pytest.raises(ValidationError):
            criterion.evaluation_schema(score=10.1, explanation="too high")

    def test_fractional_scores_work_correctly(self):
        """Fractional score values are properly validated as floats."""
        criterion = NumericalRatingCriterion(
            name="test",
            score=BoundFloat(min_val=0.0, max_val=10.0),
            explanation=TextField()
        )

        evaluation = criterion.evaluation_schema(score=7.5, explanation="good")

        assert evaluation.score == 7.5
        assert isinstance(evaluation.score, float)


class TestCritic:
    """Tests for Critic evaluation executor."""

    def test_forward_returns_evaluation_matching_schema(self):
        """Critic.forward() returns instance of criterion.evaluation_schema."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        result = critic.forward(output="test output")

        assert isinstance(result, criterion.evaluation_schema)
        assert result.passed is True
        assert result.reason == "good"

    def test_forward_passes_format_override_to_evaluator(self):
        """Critic passes criterion.evaluation_schema as format_override."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        critic.forward(output="test output")

        assert evaluator.last_prompt.format_override == criterion.evaluation_schema

    def test_prompt_template_receives_output_and_criterion_render(self):
        """Formatted prompt includes output and criterion.render()."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Criterion: {criterion}\nOutput: {output}"
        )

        critic.forward(output="test output")

        assert "test output" in evaluator.last_prompt.content
        assert "test" in evaluator.last_prompt.content

    def test_batch_forward_creates_batch_evaluation_with_list(self):
        """Critic.batch_forward() returns batch_evaluation_schema instance."""
        response_json = '{"criterion_name": "test", "evaluations": [{"criterion_name": "test", "passed": true, "reason": "good"}, {"criterion_name": "test", "passed": false, "reason": "bad"}]}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {outputs}"
        )

        result = critic.batch_forward(outputs=["output1", "output2"])

        assert isinstance(result, criterion.batch_evaluation_schema)
        assert len(result.evaluations) == 2
        assert result.evaluations[0].passed is True
        assert result.evaluations[1].passed is False

    def test_reference_parameter_is_included_in_template(self):
        """Reference argument is formatted into prompt template."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Reference: {reference}\nOutput: {output}"
        )

        critic.forward(output="test output", reference="expected output")

        assert "expected output" in evaluator.last_prompt.content

    def test_context_parameter_is_included_in_template(self):
        """Context argument is formatted into prompt template."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockProcess(response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = Critic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Context: {context}\nOutput: {output}"
        )

        critic.forward(output="test output", context={"key": "value"})

        assert "key" in evaluator.last_prompt.content or "value" in evaluator.last_prompt.content
