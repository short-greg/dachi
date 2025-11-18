import typing as t
from typing import List
from abc import abstractmethod

from pydantic import BaseModel

from dachi.inst import BaseCriterion, BatchEvaluation, Evaluation
from dachi.core import Prompt, ParamSet, Module, Msg
from ._ai import LLM
from abc import ABC
from dachi.proc import Process, AsyncProcess
from dachi.core import render, modfield


T = t.TypeVar("T", bound=Module)
L = t.TypeVar("L", bound=LLM)
C = t.TypeVar("C", bound=BaseCriterion)
P = t.TypeVar("P", bound=Process | AsyncProcess)


class Optim(Module, ABC):
    """Executes optimization using an LLM optimizer and a criterion."""

    @abstractmethod
    def step(self, evaluations: Evaluation | BatchEvaluation):
        pass

    @abstractmethod
    async def astep(self, evaluations: Evaluation | BatchEvaluation):
        pass


class LLMOptim(Optim, t.Generic[L, C]):
    """Executes optimization using an LLM optimizer and a criterion."""
    llm: L
    params: ParamSet
    criterion: C
    prompt_template: str = """
Update the parameters to optimize objective and satisfy constraints.

Objective:
{objective}

Constraints:
{constraints}
"""

    @abstractmethod
    def objective(self) -> str:
        """Specifies how to optimize according to the criterion

        Returns:
            str: The objective description
        """
        pass
        
    @abstractmethod
    def constraints(self) -> str:
        """Specifies any constraints on the optimization

        Returns:
            str: The constraints description
        """
        pass

    @abstractmethod
    def param_evaluations(self, evaluations: Evaluation | BatchEvaluation) -> str:
        """Formats the parameter evaluations for the prompt

        Args:
            evaluations (Evaluation | EvaluationBatch): The evaluations to format

        Returns:
            str: The formatted evaluations
        """
        pass
    
    def step(self, evaluations: Evaluation | BatchEvaluation):

        evaluations = self.param_evaluations(evaluations)
        objective = self.objective()
        constraints = self.constraints()
        prompt_text = self.prompt_template.format(
            objective=objective,
            constraints=constraints
        )
        system_msg = Prompt(
            role="system",
            content=prompt_text,
            format_override=self.params.schema()
        )
        user_msg = Prompt(
            role="user",
            content=render(evaluations)
        )
        updated_params = self.llm.forward(
            [system_msg, *self.thread, user_msg]
        )
        self.params.update(updated_params)

    @property
    def thread(self) -> t.Optional[t.List[Msg]]:
        return []

    async def astep(self, evaluations):
        
        evaluations = self.param_evaluations(evaluations)
        objective = self.objective()
        constraints = self.constraints()
        prompt_text = self.prompt_template.format(
            objective=objective,
            constraints=constraints
        )
        system_msg = Prompt(
            role="system",
            content=prompt_text,
            format_override=self.params.schema()
        )
        user_msg = Prompt(
            role="user",
            content=render(evaluations)
        )
        updated_params = await self.llm.aforward(
            [system_msg, *self.thread, user_msg]
        )
        self.params.update(updated_params)


class Critic(Process, AsyncProcess, t.Generic[C, P]):
    """Executes evaluations using an LLM evaluator and a criterion."""
    
    criterion: C
    evaluator: P
    prompt_template: str
    reference: t.Any | None = None

    def forward(
        self, 
        output, 
        input=None, 
        reference=None, 
        context=None, 
        **kwargs
    ) -> BaseModel:
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
