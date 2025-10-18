from __future__ import annotations

import typing
from abc import abstractmethod
from typing import Type, List
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model
from ..core import Renderable, Prompt
from ..proc import Process, AsyncProcess


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

        return create_model(
            f'{self.name.replace(" ", "_")}Evaluation',
            **fields,
            __base__=BaseModel
        )

    def _create_batch(self, single_schema: Type[BaseModel]) -> Type[BaseModel]:
        """Create batch evaluation schema."""
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            criterion_name=(str, Field(default=self.name)),
            evaluations=(List[single_schema], Field(description="List of evaluations")),
            __base__=BaseModel
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


class Critic(Process, AsyncProcess):
    """Executes evaluations using an LLM evaluator and a criterion."""

    criterion: BaseCriterion
    evaluator: Process | AsyncProcess
    prompt_template: str
    reference: typing.Any | None = None

    def forward(self, output, input=None, reference=None, context=None, **kwargs) -> BaseModel:
        """Execute single evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            output=output,
            input=input or "",
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        prompt = Prompt(
            role="user",
            content=prompt_text,
            format_override=self.criterion.evaluation_schema
        )

        resp = self.evaluator.forward(prompt)
        return self.criterion.evaluation_schema.model_validate_json(resp.text)

    async def aforward(self, output, input=None, reference=None, context=None, **kwargs) -> BaseModel:
        """Async single evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            output=output,
            input=input or "",
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        prompt = Prompt(
            role="user",
            content=prompt_text,
            format_override=self.criterion.evaluation_schema
        )

        if isinstance(self.evaluator, AsyncProcess):
            resp = await self.evaluator.aforward(prompt)
        else:
            resp = self.evaluator.forward(prompt)

        return self.criterion.evaluation_schema.model_validate_json(resp.text)

    def batch_forward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
        """Execute batch evaluation."""
        outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            outputs=outputs_text,
            output=outputs_text,
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        prompt = Prompt(
            role="user",
            content=prompt_text,
            format_override=self.criterion.batch_evaluation_schema
        )

        resp = self.evaluator.forward(prompt)
        return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)

    async def batch_aforward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
        """Async batch evaluation."""
        outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            outputs=outputs_text,
            output=outputs_text,
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        prompt = Prompt(
            role="user",
            content=prompt_text,
            format_override=self.criterion.batch_evaluation_schema
        )

        if isinstance(self.evaluator, AsyncProcess):
            resp = await self.evaluator.aforward(prompt)
        else:
            resp = self.evaluator.forward(prompt)

        return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)
