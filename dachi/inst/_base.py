from __future__ import annotations

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model
import pydantic

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model
import pydantic


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

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        fields = ', '.join(
            f'{k}={getattr(self, k)}' for k, _ in self.__class__.model_fields.items()
        )
        return f'{cls_name}({fields})'


class BaseCriterion(BaseModel):
    """Base class for all criteria. Auto-generates evaluation schemas from EvalFields."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None

    _evaluation_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)
    _batch_evaluation_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)

    @property
    def evaluation_schema(self) -> t.Type[BaseModel]:
        """Get the single evaluation schema."""
        if self._evaluation_schema is None:
            raise RuntimeError("evaluation_schema not initialized")
        return self._evaluation_schema

    @property
    def batch_evaluation_schema(self) -> t.Type[BaseModel]:
        """Get the batch evaluation schema."""
        if self._batch_evaluation_schema is None:
            raise RuntimeError("batch_evaluation_schema not initialized")
        return self._batch_evaluation_schema

    def model_post_init(self, __context) -> None:
        """Auto-generate evaluation schemas from EvalFields."""
        super().model_post_init(__context)
        single = self._create_single()
        batch = self._create_batch(single)

        object.__setattr__(self, '_evaluation_schema', single)
        object.__setattr__(self, '_batch_evaluation_schema', batch)

    def _create_single(self) -> t.Type[BaseModel]:
        """Create single evaluation schema by introspecting EvalFields.

        Note: criterion_name uses Optional[str] with default to support OpenAI strict mode.
        OpenAI strict mode requires all properties in 'required' array but doesn't support
        traditional defaults. Instead, we use nullable types (str | None) and apply the
        default after validation if the LLM returns null.
        """
        fields = {
            # #'criterion_name': (
            #     t.Optional[str],
            #     Field(default=self.name, description="Name of the criterion")
            # )
        }

        # Use model_fields to find EvalField annotations
        for field_name, field_info in self.__class__.model_fields.items():
            field_value = getattr(self, field_name)
            if isinstance(field_value, EvalField):
                fields[field_name] = field_value.get_field()
            else:
                
                fields[field_name] = (t.Literal[field_value], Field(default=field_value))
        # include eval_type in the model that is 
        # "single" or "batch"
        return create_model(
            f'{self.name.replace(" ", "_")}Evaluation',
            **fields,
            __base__=Evaluation,
        )

    def _create_batch(self, single_schema: t.Type[BaseModel]) -> t.Type[BaseModel]:
        """Create batch evaluation schema.

        Note: criterion_name uses Optional[str] with default to support OpenAI strict mode.
        """
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            # criterion_name=(t.Optional[str], Field(default=self.name, description="Name of the criterion")),
            evaluations=(t.List[single_schema], Field(description="List of evaluations")),
            __base__=BatchEvaluation
        )
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        fields = ', '.join(
            f'{k}={getattr(self, k)}' for k, _ in self.__class__.model_fields.items()
        )
        return f'{cls_name}({fields})'


CRITERION = t.TypeVar("CRITERION", bound=BaseCriterion)


class CriterionMixin(BaseModel):
    """Mixin for criteria with reasoning."""
    pass


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


class Evaluation(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    model_config = ConfigDict(extra='forbid')

    def to_record(self) -> t.Dict:
        """
        Convert the evaluation to a record.
        Returns:
            t.Dict: A record
        """
        return self.model_dump()


class BatchEvaluation(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    model_config = ConfigDict(extra='forbid')
    evaluations: t.List[Evaluation]

    def to_records(self) -> t.List[t.Dict]:
        """
        Convert the evaluations to a list of records.
        Returns:
            t.List[t.Dict]: A list of records
        """
        return [
            self.evaluations[i].to_record()
            for i in self.evaluations.keys()
        ]
