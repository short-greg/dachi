import typing
import pytest

from dachi.inst._criterion import (
    PassFailCriterion,
    TextField,
    BoolField,
)
from dachi.proc._optim import LangCritic
from dachi.proc._ai import LangModel
from dachi.core import Inp

from dachi.proc import LangOptim, LangModel
from dachi.core import ParamSet, Param, TextMsg
from dachi.inst import BaseCriterion, BoundInt, Evaluation, BatchEvaluation
from pydantic import Field

class MockLangModel(LangModel):
    """Mock LangModel for testing LangCritic."""

    response_json: str
    last_prompt: str = None
    last_structure: typing.Any = None

    def forward(self, prompt, structure=None, tools=None, **kwargs) -> typing.Tuple[str, typing.List[Inp], typing.Any]:
        self.last_prompt = prompt
        self.last_structure = structure
        return (self.response_json, [], None)

    async def aforward(self, prompt, structure=None, tools=None, **kwargs) -> typing.Tuple[str, typing.List[Inp], typing.Any]:
        self.last_prompt = prompt
        self.last_structure = structure
        return (self.response_json, [], None)

    def stream(self, prompt, structure=None, tools=None, **kwargs) -> typing.Iterator[typing.Tuple[str, typing.List[Inp], typing.Any]]:
        self.last_prompt = prompt
        self.last_structure = structure
        yield (self.response_json, [], None)

    async def astream(self, prompt, structure=None, tools=None, **kwargs) -> typing.AsyncIterator[typing.Tuple[str, typing.List[Inp], typing.Any]]:
        self.last_prompt = prompt
        self.last_structure = structure
        yield (self.response_json, [], None)


class TestCritic:
    """Tests for Critic evaluation executor."""

    def test_forward_returns_evaluation_matching_schema(self):
        """LangCritic.forward() returns instance of criterion.evaluation_schema."""
        response_json = '{"name": "test", "passed": true, "passing_criteria": "criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        result = critic.forward(output="test output")

        assert isinstance(result, criterion.evaluation_schema)
        assert result.passed is True

    def test_forward_passes_structure_to_evaluator(self):
        """LangCritic passes criterion.evaluation_schema as structure."""
        response_json = '{"name": "test", "passed": true, "passing_criteria": "criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
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
        response_json = '{"name": "test", "passed": true, "passing_criteria": "criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
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
        response_json = '{"evaluations": [{"name": "test", "passed": true, "passing_criteria": "passed"}, {"name": "test", "passed": false, "passing_criteria": "failed"}]}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
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
        response_json = '{"name": "test", "passed": true, "passing_criteria": "criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
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
        response_json = '{"name": "test", "passed": true, "passing_criteria": "criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField(),
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Context: {context}\nOutput: {output}"
        )

        critic.forward(output="test output", context={"key": "value"})

        assert "key" in evaluator.last_prompt or "value" in evaluator.last_prompt

    @pytest.mark.asyncio
    async def test_aforward_returns_evaluation_matching_schema(self):
        """LangCritic.aforward() returns instance of criterion.evaluation_schema."""
        response_json = '{"name": "test", "passed": true, "passing_criteria": "async criteria was met"}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {output}"
        )

        result = await critic.aforward(output="test output")

        assert isinstance(result, criterion.evaluation_schema)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_batch_aforward_creates_batch_evaluation_with_list(self):
        """LangCritic.batch_aforward() returns batch_evaluation_schema instance."""
        response_json = '{"evaluations": [{"name": "test", "passed": true, "passing_criteria": "passed"}, {"name": "test", "passed": false, "passing_criteria": "failed"}]}'
        evaluator = MockLangModel(response_json=response_json)

        criterion = PassFailCriterion(
            name="test",
            passed=BoolField()
        )

        critic = LangCritic(
            criterion=criterion,
            evaluator=evaluator,
            prompt_template="Evaluate: {outputs}"
        )

        result = await critic.batch_aforward(outputs=["output1", "output2"])

        assert isinstance(result, criterion.batch_evaluation_schema)
        assert len(result.evaluations) == 2
        assert result.evaluations[0].passed is True
        assert result.evaluations[1].passed is False


"""Test LangOptim subclassing to diagnose validation error."""


class SimpleCriterion(BaseCriterion):
    """Minimal criterion for testing."""

    name: str = "Test"
    description: str = "Test criterion"
    score: BoundInt = Field(
        default_factory=lambda: BoundInt(min_val=1, max_val=5, description="Score")
    )


class SimpleLangOptim(LangOptim):
    """Minimal LangOptim subclass to test inheritance."""

    objective_text: str
    constraints_text: str

    def objective(self) -> str:
        return self.objective_text

    def constraints(self) -> str:
        return self.constraints_text

    def param_evaluations(self, evaluations: Evaluation | BatchEvaluation) -> str:
        return str(evaluations)


def test_langoptim_instantiation():
    """Test that LangOptim subclass can be instantiated."""
    print("Creating MockLangModel...")
    llm = MockLangModel(response_json='{}')
    print(f"  Success: {llm}")

    print("Creating SimpleCriterion...")
    criterion = SimpleCriterion()
    print(f"  Success: {criterion}")

    print("Creating ParamSet...")
    params = ParamSet(params=[Param(data="test prompt")])
    print(f"  Success: {params}")

    print("Creating SimpleLangOptim...")
    try:
        optimizer = SimpleLangOptim(
            llm=llm,
            params=params,
            criterion=criterion,
            objective_text="Test objective",
            constraints_text="Test constraints",
            prompt_template="Objective: {objective}\nConstraints: {constraints}"
        )
        print(f"  Success: {optimizer}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        raise


def test_langoptim_step_updates_params():
    """Test that LangOptim.step() properly updates parameters."""
    llm = MockLangModel(response_json='{"param_0": {"data": "updated prompt"}}')
    criterion = SimpleCriterion()
    params = ParamSet(params=[Param(data="initial prompt")])

    optimizer = SimpleLangOptim(
        llm=llm,
        params=params,
        criterion=criterion,
        objective_text="Maximize effectiveness",
        constraints_text="Keep it concise",
        prompt_template="Objective: {objective}\nConstraints: {constraints}"
    )

    evaluation = criterion.evaluation_schema(score=3)
    optimizer.step(evaluation)

    assert optimizer.params.params[0].data == "updated prompt"


@pytest.mark.asyncio
async def test_langoptim_astep_updates_params():
    """Test that LangOptim.astep() properly updates parameters asynchronously."""
    llm = MockLangModel(response_json='{"param_0": {"data": "async updated prompt"}}')
    criterion = SimpleCriterion()
    params = ParamSet(params=[Param(data="initial prompt")])

    optimizer = SimpleLangOptim(
        llm=llm,
        params=params,
        criterion=criterion,
        objective_text="Maximize effectiveness",
        constraints_text="Keep it concise",
        prompt_template="Objective: {objective}\nConstraints: {constraints}"
    )

    evaluation = criterion.evaluation_schema(score=3)
    await optimizer.astep(evaluation)

    assert optimizer.params.params[0].data == "async updated prompt"


def test_langoptim_thread_property_returns_empty_list():
    """Test that LangOptim.thread property returns empty list."""
    llm = MockLangModel(response_json='{}')
    criterion = SimpleCriterion()
    params = ParamSet(params=[Param(data="test")])

    optimizer = SimpleLangOptim(
        llm=llm,
        params=params,
        criterion=criterion,
        objective_text="Test",
        constraints_text="Test",
        prompt_template="Test"
    )

    assert optimizer.thread == []
    assert isinstance(optimizer.thread, list)


if __name__ == "__main__":
    test_langoptim_instantiation()
    print("\nAll tests passed!")

