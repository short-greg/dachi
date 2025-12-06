import typing

from dachi.inst._criterion import (
    PassFailCriterion,
    TextField,
    BoolField,
)
from dachi.proc._optim import LangCritic
from dachi.proc._ai import LangModel
from dachi.core import Inp


class MockLangModel(LangModel):
    """Mock LangModel for testing LangCritic."""

    response_json: str
    last_prompt: str = None
    last_structure: typing.Any = None

    def forward(self, prompt, structure=None, tools=None, **kwargs) -> typing.Tuple[str, typing.List[Inp]]:
        self.last_prompt = prompt
        self.last_structure = structure
        return (self.response_json, [])

    async def aforward(self, prompt, structure=None, tools=None, **kwargs) -> typing.Tuple[str, typing.List[Inp]]:
        self.last_prompt = prompt
        self.last_structure = structure
        return (self.response_json, [])

    def stream(self, prompt, structure=None, tools=None, **kwargs) -> typing.Iterator[typing.Tuple[str, typing.List[Inp]]]:
        self.last_prompt = prompt
        self.last_structure = structure
        yield (self.response_json, [])

    async def astream(self, prompt, structure=None, tools=None, **kwargs) -> typing.AsyncIterator[typing.Tuple[str, typing.List[Inp]]]:
        self.last_prompt = prompt
        self.last_structure = structure
        yield (self.response_json, [])


class TestCritic:
    """Tests for Critic evaluation executor."""

    def test_forward_returns_evaluation_matching_schema(self):
        """LangCritic.forward() returns instance of criterion.evaluation_schema."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        result = critic.forward(output="test output")

        assert isinstance(result, criterion.evaluation_schema)
        assert result.passed is True
        assert result.reason == "good"

    def test_forward_passes_structure_to_evaluator(self):
        """LangCritic passes criterion.evaluation_schema as structure."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        critic.forward(output="test output")

        assert evaluator.last_structure == criterion.evaluation_schema

    def test_prompt_template_receives_output_and_criterion_render(self):
        """Formatted prompt includes output and criterion.render()."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Criterion: {criterion}\nOutput: {output}"
        )

        critic.forward(output="test output")

        assert "test output" in evaluator.last_prompt
        assert "test" in evaluator.last_prompt

    def test_batch_forward_creates_batch_evaluation_with_list(self):
        """LangCritic.batch_forward() returns batch_evaluation_schema instance."""
        response_json = '{"criterion_name": "test", "evaluations": [{"criterion_name": "test", "passed": true, "reason": "good"}, {"criterion_name": "test", "passed": false, "reason": "bad"}]}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
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
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Reference: {reference}\nOutput: {output}"
        )

        critic.forward(output="test output", reference="expected output")

        assert "expected output" in evaluator.last_prompt

    def test_context_parameter_is_included_in_template(self):
        """Context argument is formatted into prompt template."""
        response_json = '{"criterion_name": "test", "passed": true, "reason": "good"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
            reason=TextField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Context: {context}\nOutput: {output}"
        )

        critic.forward(output="test output", context={"key": "value"})

        assert "key" in evaluator.last_prompt or "value" in evaluator.last_prompt
