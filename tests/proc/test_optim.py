from dachi.inst._criterion import (
    PassFailCriterion,
    TextField,
    BoolField,
)
from dachi.proc._optim import Critic
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
