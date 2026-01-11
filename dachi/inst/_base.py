from __future__ import annotations

import typing as t
from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, create_model, model_validator

import typing as t
from abc import abstractmethod


class RespField(BaseModel):
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


class ResponseSpec(BaseModel):
    """Base class for all criteria. Auto-generates evaluation schemas from EvalFields."""

    model_config = ConfigDict(frozen=True)
    name: str
    description: str | None = None

    def model_post_init(self, __context):
        """Post-init to create response schemas."""
        super().model_post_init(__context)
        single_schema = self._create_single()
        batch_schema = self._create_batch(single_schema)
        self._response_schema = single_schema
        self._batch_response_schema = batch_schema

    _response_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)
    _batch_response_schema: t.Type[BaseModel] | None = PrivateAttr(default=None)

    @model_validator(mode='before')
    @classmethod
    def validate_field(cls, values: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Convert any field that has been passed in as a string."""
        for field_name, field_info in cls.model_fields.items():
            try:
                if issubclass(field_info.annotation, RespField):
                    value = values.get(field_name)
                    if value is not None and isinstance(value, str):
                        # Allow string shorthand for RespField
                        values[field_name] = field_info.annotation(description=value)
            except TypeError:
                # Skip non-class annotations (e.g., Union types like str | None)
                continue

        return values

    @property
    def response_schema(self) -> t.Type[BaseModel] | None:
        """Get the single evaluation schema."""
        return self._response_schema

    @property
    def batch_response_schema(self) -> t.Type[BaseModel] | None:
        """Get the batch evaluation schema."""
        return self._batch_response_schema

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
            if isinstance(field_value, RespField):
                fields[field_name] = field_value.get_field()
            else:
                
                fields[field_name] = (t.Literal[field_value], Field(default=field_value))
        # include eval_type in the model that is 
        # "single" or "batch"
        return create_model(
            f'{self.name.replace(" ", "_")}Response',
            **fields,
            __base__=BaseResponse,
        )

    def _create_batch(self, single_schema: t.Type[BaseModel]) -> t.Type[BaseModel]:
        """Create batch evaluation schema.

        Note: criterion_name uses Optional[str] with default to support OpenAI strict mode.
        """
        return create_model(
            f'{self.name.replace(" ", "_")}BatchEvaluation',
            # criterion_name=(t.Optional[str], Field(default=self.name, description="Name of the criterion")),
            responses=(t.List[single_schema], Field(description="List of responses")),
            __base__=BaseBatchResponse
        )
    
    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        fields = ', '.join(
            f'{k}={getattr(self, k)}' for k, _ in self.__class__.model_fields.items()
        )
        return f'{cls_name}({fields})'


RESPONSE_SPEC = t.TypeVar("RESPONSE_SPEC", bound=ResponseSpec)


class RespField(BaseModel):
    """Base class for evaluation field descriptors."""

    description: str | None = None

    @abstractmethod
    def get_field(self) -> tuple:
        """Return (type, Field(...)) tuple for create_model.

        Returns:
            tuple: (field_type, Field(...)) for use in create_model
        """
        pass


def to_records(responses: t.List['BaseResponse'] | 'BaseBatchResponse') -> t.List[t.Dict]:
    """Convert response(s) to list of records.

    Args:
        response: Single or batch response

    Returns:
        t.List[t.Dict]: List of records
    """
    if isinstance(responses, BaseBatchResponse):
        responses = responses.evaluations

    records = []
    for response in responses:
        if hasattr(response, 'to_record'):
            records.append(response.to_record())
        else:
            records.append(response.model_dump())

    return records
    

class BaseResponse(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    model_config = ConfigDict(extra='forbid')


class BaseBatchResponse(BaseModel):
    """
    A evaluation is a function that takes in a set of parameters and returns a value.
    This value is used to evaluate the performance of the parameters.
    """
    model_config = ConfigDict(extra='forbid')
    responses: t.List[BaseResponse]

    def to_records(self) -> t.List[t.Dict]:
        """
        Convert the evaluations to a list of records.
        Returns:
            t.List[t.Dict]: A list of records
        """
        records = []
        for response in self.responses:
            if hasattr(response, 'to_record'):
                records.append(response.to_record())
            else:
                records.append(response.model_dump())
        return records
