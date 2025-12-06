import typing as t
from typing import List
from abc import abstractmethod

from pydantic import BaseModel

from dachi.inst import CRITERION, BatchEvaluation, Evaluation
from dachi.core import ParamSet, Module, Inp
from abc import ABC
from dachi.proc import Process, AsyncProcess, PROCESS
from dachi.core import render, TextMsg

from ._ai import LangModel, LANG_MODEL
from ._inst import TemplateFormatter


class Optim(Module, ABC):
    """Executes optimization using an LLM optimizer and a criterion."""

    @abstractmethod
    def step(self, evaluations: Evaluation | BatchEvaluation):
        pass

    @abstractmethod
    async def astep(self, evaluations: Evaluation | BatchEvaluation):
        pass


class LangOptim(Optim, t.Generic[CRITERION]):
    """Executes optimization using an LLM optimizer and a criterion."""
    llm: LangModel
    params: ParamSet
    criterion: CRITERION
    prompt_template: str

    def model_post_init(self, __context):
        self.formatter = TemplateFormatter()

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
        system = TextMsg("system", prompt_text)
        user = TextMsg("user", render(evaluations))

        updated_params = self.llm.forward(
            [system, user], structure=self.params.to_schema()
        )
        self.params.update(updated_params)

    @property
    def thread(self) -> t.Optional[t.List[Inp]]:
        return []

    async def astep(self, evaluations):
        
        evaluations = self.param_evaluations(evaluations)
        objective = self.objective()
        constraints = self.constraints()
        prompt_text = self.prompt_template.format(
            objective=objective,
            constraints=constraints
        )
        system = TextMsg("system", prompt_text)
        user = TextMsg("user", render(evaluations))

        updated_params = await self.llm.aforward(
            [system, user], structure=self.params.to_schema()
        )
        self.params.update(updated_params)


class LangCritic(Process, AsyncProcess, t.Generic[CRITERION, LANG_MODEL]):
    """Executes evaluations using an LLM evaluator and a criterion."""

    criterion: CRITERION
    evaluator: LANG_MODEL
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

        text, _ = self.evaluator.forward(
            prompt_text, structure=self.criterion.evaluation_schema
        )
        return self.criterion.evaluation_schema.model_validate_json(text)

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


        if isinstance(self.evaluator, AsyncProcess):
            text, _ = await self.evaluator.aforward(
                prompt_text, structure=self.criterion.evaluation_schema
            )
        else:
            text, _ = self.evaluator.forward(prompt_text, structure=self.criterion.evaluation_schema)
        return self.criterion.evaluation_schema.model_validate_json(text)
    
    def batch_forward(
        self, outputs: List,
        inputs: List = None,
        reference=None, context=None, **kwargs
    ) -> BaseModel:
        """Execute batch evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            outputs=outputs,
            inputs=inputs or [],
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )


        text, _ = self.evaluator.forward(
            prompt_text, structure=self.criterion.batch_evaluation_schema
        )
        return self.criterion.batch_evaluation_schema.model_validate_json(text)
    
    async def batch_aforward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
        """Async batch evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=self.criterion.render(),
            outputs=outputs,
            inputs=inputs or [],
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        if isinstance(self.evaluator, AsyncProcess):
            text, _ = await self.evaluator.aforward(prompt_text, structure=self.criterion.batch_evaluation_schema)
        else:
            text, _ = self.evaluator.forward(prompt_text, structure=self.criterion.batch_evaluation_schema)

        return self.criterion.batch_evaluation_schema.model_validate_json(text)
    

        
    #     outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

    #     prompt_text = self.prompt_template.format(
    #         criterion=self.criterion.render(),
    #         outputs=outputs_text,
    #         output=outputs_text,
    #         reference=reference or self.reference or "",
    #         context=context or {},
    #         **kwargs
    #     )

    #     prompt = Inp(
    #         role="user",
    #         content=prompt_text,
    #         format_override=self.criterion.batch_evaluation_schema
    #     )

    #     resp = self.evaluator.forward(prompt)
    #     return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)

    # async def batch_aforward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
    #     """Async batch evaluation."""
    #     outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

    #     prompt_text = self.prompt_template.format(
    #         criterion=self.criterion.render(),
    #         outputs=outputs_text,
    #         output=outputs_text,
    #         reference=reference or self.reference or "",
    #         context=context or {},
    #         **kwargs
    #     )

    #     prompt = Inp(
    #         role="user",
    #         content=prompt_text,
    #         format_override=self.criterion.batch_evaluation_schema
    #     )

    #     if isinstance(self.evaluator, AsyncProcess):
    #         resp = await self.evaluator.aforward(prompt)
    #     else:
    #         resp = self.evaluator.forward(prompt)

    #     return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)
    
    # def batch_forward(
    #     self, outputs: List, 
    #     inputs: List = None, 
    #     reference=None, context=None, **kwargs
    # ) -> BaseModel:
    # """Execute batch evaluation."""
    # outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

    # prompt_text = self.prompt_template.format(
    #     criterion=self.criterion.render(),
    #     outputs=outputs_text,
    #     output=outputs_text,
    #     reference=reference or self.reference or "",
    #     context=context or {},
    #     **kwargs
    # )

    # prompt = Prompt(
    #     role="user",
    #     content=prompt_text,
    #     format_override=self.criterion.batch_evaluation_schema
    # )

    # resp = self.evaluator.forward(prompt)
    # return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)

    # async def batch_aforward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
    #     """Async batch evaluation."""
    #     outputs_text = "\n\n".join(f"Output {i+1}:\n{out}" for i, out in enumerate(outputs))

    #     prompt_text = self.prompt_template.format(
    #         criterion=self.criterion.render(),
    #         outputs=outputs_text,
    #         output=outputs_text,
    #         reference=reference or self.reference or "",
    #         context=context or {},
    #         **kwargs
    #     )

    #     prompt = Prompt(
    #         role="user",
    #         content=prompt_text,
    #         format_override=self.criterion.batch_evaluation_schema
    #     )

    #     if isinstance(self.evaluator, AsyncProcess):
    #         resp = await self.evaluator.aforward(prompt)
    #     else:
    #         resp = self.evaluator.forward(prompt)

    #     return self.criterion.batch_evaluation_schema.model_validate_json(resp.text)
