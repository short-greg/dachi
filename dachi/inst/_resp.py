from __future__ import annotations

from pydantic import Field
from ._field import (
    BoundFloat, BoolField, 
    TextField, ListField, FixedListField
)
from ._base import ResponseSpec


class Reason(ResponseSpec):
    """Reasoning text field."""
    reason: TextField = Field(
        default_factory=lambda: TextField(description="The reason for the judgment")
    )


class Brainstorming(ResponseSpec):
    """Brainstorming text field."""
    ideas: ListField = Field(
        default_factory=lambda: ListField(item_type=str, description="List of brainstorming ideas")
    )


class Plan(ResponseSpec):
    """Planning text field."""
    plan: ListField = Field(
        default_factory=lambda: ListField(item_type=str, description="The detailed plan of action")
    )


class Hypothesis(ResponseSpec):
    """Hypothesis text field."""
    hypothesis: TextField = Field(
        default_factory=lambda: TextField(description="A hypothesis to be tested and evaluated.")
    )


class Sample(ResponseSpec):
    """Sample text field."""
    sample: TextField = Field(
        default_factory=lambda: TextField(description="The provided sample")
    )


class Probability(ResponseSpec):
    """Probability float field."""
    probability: BoundFloat = Field(
        default_factory=lambda: BoundFloat(description="The subjective probability.", min_val=0.0, max_val=1.0)
    )


class Confidence(ResponseSpec):
    """Probability float field."""
    probability: BoundFloat = Field(
        default_factory=lambda: BoundFloat(description="The confidence level for the item.", min_val=0.0, max_val=1.0)
    )


class HypoSupport(ResponseSpec):
    """Hypothesis support field."""
    support: BoolField = Field(
        default_factory=lambda: BoolField(description="Whether the hypothesis is supported by the evidence")
    )  


class Evidence(ResponseSpec):
    """Hypothesis evaluation field."""
    evidence: ListField = Field(
        default_factory=lambda: ListField(item_type=str, description="The evidence supporting or refuting the hypothesis.")
    )


class ListResp(ResponseSpec):
    """Batch response field."""
    responses: ListField = Field(
        item_type=dict,
        default_factory=lambda: ListField(item_type=dict, description="List of responses for each item in the batch")
    )


class InstructionSet(ResponseSpec):
    """Instruction set field."""
    instructions: FixedListField = Field(
        n_items=3,
        default_factory=lambda: FixedListField(item_type=str, n_items=3, description="Set of instructions to follow.")
    )
