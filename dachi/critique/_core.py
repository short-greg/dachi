# - Critic
#     - Name
#     - Schema
# - EvaluationSchema
#     - Name
#     - Description
#     - instruction()
#     - out_schema()
#     - Use a particular schema for the evaluation
#     - {‘name’: string | other type}
# - Likert(Schema) ⇒
#     - scales
#     - pass in the names of the
# - CompositeSchema(Schema)
#     - out()
# - critique(likert, …)
# - batch_critique(likert, …)
# - Critic(Module) ⇒ Evaluation
#     - Schema
# - BatchCritic(Module) ⇒ BatchEvaluation
#     - Get the instruction from the critic. Update the output instruction… Perhaps I just need a “batch” option..  BatchReader.. Add the batch reader
# - Batch (an array of data)

from abc import ABC, abstractmethod
from typing import Any
import typing
from .._core import Module, Struct, Instruction


class Evaluation(Struct):
    pass


class CompositeEvaluation(Evaluation):
    pass


class EvaluationSchema(Struct, ABC):

    name: str

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def instruction(self) -> Instruction:
        pass

    @abstractmethod
    def out_schema(self) -> typing.Dict:
        pass


class Likert(EvaluationSchema):
    pass


class Copmosite(EvaluationSchema):
    pass


def batch_critique():
    pass


def critique():
    pass


class Critic(Module):

    def __init__(self, schema):
        super().__init__()
        self.schema = schema

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass


class BatchCritic(Module):

    def __init__(self, schema):
        super().__init__()
        self.schema = schema

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
