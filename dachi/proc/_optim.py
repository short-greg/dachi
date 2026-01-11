import typing as t
from typing import List
from abc import abstractmethod
import json

import pydantic
from pydantic import BaseModel

from dachi.inst import RESPONSE_SPEC, BaseBatchResponse, BaseResponse
from dachi.core import ParamSet, Module, Inp
from abc import ABC
from dachi.proc import Process, AsyncProcess
from dachi.core import TextMsg

from ._lang import LangModel, LANG_MODEL
from ._inst import TemplateFormatter


class Optim(Module):
    """Executes optimization using an LLM optimizer and a criterion."""

    @abstractmethod
    def step(self, evaluations: BaseResponse | BaseBatchResponse):
        raise NotImplementedError

    @abstractmethod
    async def astep(self, evaluations: BaseResponse | BaseBatchResponse):
        raise NotImplementedError


class LangOptim(Optim, t.Generic[RESPONSE_SPEC]):
    """Executes optimization using an LLM optimizer and a criterion."""
    llm: LangModel
    params: ParamSet
    criterion: RESPONSE_SPEC
    prompt_template: str
    _formatter: TemplateFormatter = pydantic.PrivateAttr()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._formatter = TemplateFormatter(prompt_template=self.prompt_template)

    @abstractmethod
    def objective(self) -> str:
        """Specifies how to optimize according to the criterion

        Returns:
            str: The objective description
        """
        raise NotImplementedError
        
    @abstractmethod
    def constraints(self) -> str:
        """Specifies any constraints on the optimization

        Returns:
            str: The constraints description
        """
        raise NotImplementedError

    @abstractmethod
    def param_evaluations(self, evaluations: BaseResponse | BaseBatchResponse) -> str:
        """Formats the parameter evaluations for the prompt

        Args:
            evaluations (Evaluation | EvaluationBatch): The evaluations to format

        Returns:
            str: The formatted evaluations
        """
        raise NotImplementedError
    
    def step(self, evaluations: BaseResponse | BaseBatchResponse):

        evaluations = self.param_evaluations(evaluations)
        objective = self.objective()
        constraints = self.constraints()
        prompt_text = self.prompt_template.format(
            objective=objective,
            constraints=constraints
        )
        system = TextMsg("system", prompt_text)
        user = TextMsg("user", str(evaluations))

        text, _, _ = self.llm.forward(
            [system, user], structure=self.params.to_schema()
        )
        self.params.update(json.loads(text))

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
        user = TextMsg("user", str(evaluations))

        text, _, _ = await self.llm.aforward(
            [system, user], structure=self.params.to_schema()
        )
        self.params.update(json.loads(text))


class LangCritic(Process, AsyncProcess, t.Generic[RESPONSE_SPEC, LANG_MODEL]):
    """Executes evaluations using an LLM evaluator and a criterion."""

    criterion: RESPONSE_SPEC
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
            criterion=str(self.criterion),
            output=output,
            input=input or "",
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        text, _, _ = self.evaluator.forward(
            prompt_text, structure=self.criterion.response_schema
        )
        return self.criterion.response_schema.model_validate_json(text)

    async def aforward(self, output, input=None, reference=None, context=None, **kwargs) -> BaseModel:
        """Async single evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=str(self.criterion),
            output=output,
            input=input or "",
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        if isinstance(self.evaluator, AsyncProcess):
            text, _, _ = await self.evaluator.aforward(
                prompt_text, structure=self.criterion.response_schema
            )
        else:
            text, _, _ = self.evaluator.forward(prompt_text, structure=self.criterion.response_schema)
        return self.criterion.response_schema.model_validate_json(text)
    
    def batch_forward(
        self, outputs: List,
        inputs: List = None,
        reference=None, context=None, **kwargs
    ) -> BaseModel:
        """Execute batch evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=str(self.criterion),
            outputs=outputs,
            inputs=inputs or [],
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        text, _, _ = self.evaluator.forward(
            prompt_text, structure=self.criterion.batch_response_schema
        )
        return self.criterion.batch_response_schema.model_validate_json(text)
    
    async def batch_aforward(self, outputs: List, inputs: List = None, reference=None, context=None, **kwargs) -> BaseModel:
        """Async batch evaluation."""
        prompt_text = self.prompt_template.format(
            criterion=str(self.criterion),
            outputs=outputs,
            inputs=inputs or [],
            reference=reference or self.reference or "",
            context=context or {},
            **kwargs
        )

        if isinstance(self.evaluator, AsyncProcess):
            text, _, _ = await self.evaluator.aforward(prompt_text, structure=self.criterion.batch_response_schema)
        else:
            text, _, _ = self.evaluator.forward(prompt_text, structure=self.criterion.batch_response_schema)

        return self.criterion.batch_response_schema.model_validate_json(text)


class LangCalibrator(Process, AsyncProcess, t.Generic[RESPONSE_SPEC, LANG_MODEL]):
    """Use to calibrate the batches of evaluations.."""
    calibrator: LANG_MODEL
    CRITERION: RESPONSE_SPEC
    prompt_template: str
    reference: t.Any | None = None

    def forward(
        self,
        outputs: List,
        evaluations: List,
        inputs: List = None,
        reference: List = None,
        context=None,
        **kwargs
    ) -> BaseModel:
        """Execute calibration."""

        reference = reference or self.reference
        prompt_text = self.prompt_template.format(
            criterion=str(self.CRITERION),
            outputs=outputs,
            evaluations=evaluations,
            inputs=inputs or [],
            reference=reference or [],
            context=context or {},
            **kwargs
        )

        text, _, _ = self.calibrator.forward(
            prompt_text, structure=self.CRITERION.batch_response_schema
        )
        return self.CRITERION.batch_response_schema.model_validate_json(text)
    
    async def aforward(
        self,
        outputs: List,
        evaluations: List,
        inputs: List = None,
        reference: List = None,
        context=None,
        **kwargs
    ) -> BaseModel:
        """Async calibration."""

        reference = reference or self.reference
        prompt_text = self.prompt_template.format(
            criterion=str(self.CRITERION),
            outputs=outputs,
            evaluations=evaluations,
            inputs=inputs or [],
            reference=reference or [],
            context=context or {},
            **kwargs
        )

        if isinstance(self.calibrator, AsyncProcess):
            text, _, _ = await self.calibrator.aforward(
                prompt_text, structure=self.CRITERION.batch_response_schema
            )
        else:
            text, _, _ = self.calibrator.forward(
                prompt_text, structure=self.CRITERION.batch_response_schema
            )
        return self.CRITERION.batch_response_schema.model_validate_json(text)
