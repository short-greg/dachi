"""Unit tests for dachi.inst._critique

This suite covers validation, rendering and conversion helpers for the
Criterion / Evaluation family.  All tests are black‑box and keep one
assertion per test (except when logically inseparable).
"""

import typing

import pytest
import pydantic

from dachi.inst._critique import (
    Criterion,
    CompoundCriterion,
    LikertItem,
    LikertScaleCriterion,
    Evaluation,
    EvaluationBatch,
)


class TestCriterion:
    """Tests for `Criterion`."""

    def test_render_happy_path(self):
        c = Criterion(name="conciseness", description="Is the answer concise?")
        expected = {
            "name": "conciseness",
            "description": "Is the answer concise?",
            "type": str,
        }
        assert c.render() == str(expected)

    def test_type_override(self):
        c = Criterion(name="tokens", description="Number of tokens", type_=int)
        assert c.type_ is int

    def test_validation_missing_fields(self):
        with pytest.raises(pydantic.ValidationError):
            Criterion()


class TestCompoundCriterion:
    """Tests for `CompoundCriterion`."""

    def test_render_aggregates_children(self):
        c1 = Criterion(name="fluency", description="Fluent English")
        c2 = Criterion(name="style", description="Appropriate style")
        comp = CompoundCriterion(criteria=[c1, c2])
        rendered = comp.render()
        assert c1.render() in rendered and c2.render() in rendered

    def test_empty_list_allowed(self):
        comp = CompoundCriterion(criteria=[])
        assert comp.render() == str({"criteria": []})

    def test_validation_non_list(self):
        with pytest.raises(pydantic.ValidationError):
            CompoundCriterion(criteria=Criterion(name="a", description="b"))


class TestLikertItem:
    """Tests for `LikertItem`."""

    def test_valid_instance(self):
        item = LikertItem(description="Strongly agree", val=5)
        assert item.val == 5

    def test_validation_non_int_val(self):
        item = LikertItem(description="Agree", val="5")
        assert type(item.val) is int


class TestLikertScaleCriterion:
    """Tests for `LikertScaleCriterion`."""

    def test_validation_valid_scale(self):
        scale = [LikertItem(description=str(i), val=i) for i in range(1, 6)]
        lsc = LikertScaleCriterion(name="satisfaction", description="Overall satisfaction", scale=scale)
        assert len(lsc.scale) == 5

    def test_validation_non_list_scale(self):
        with pytest.raises(pydantic.ValidationError):
            LikertScaleCriterion(name="x", description="y", scale="not a list")


class TestEvaluation:
    """Tests for `Evaluation`."""

    def test_to_record_and_render(self):
        ev = Evaluation(val={"conciseness": 0.9, "fluency": "good"})
        assert ev.to_record() == {"conciseness": 0.9, "fluency": "good"}
        assert ev.render() == str(ev.val)

    def test_validation_unsupported_type(self):
        with pytest.raises(pydantic.ValidationError):
            Evaluation(val={"foo": [1, 2, 3]})


class TestEvaluationBatch:
    """Tests for `EvaluationBatch`."""

    def test_to_records_order_preserved(self):
        ev0 = Evaluation(val={"a": 1})
        ev1 = Evaluation(val={"b": 2})
        batch = EvaluationBatch(evaluations={0: ev0, 1: ev1})
        records = batch.to_records()
        assert records == [ev0.to_record(), ev1.to_record()]

    def test_validation_non_evaluation_values(self):
        with pytest.raises(pydantic.ValidationError):
            EvaluationBatch(evaluations={0: {"not": "evaluation"}})


# class TestRenderableContract:
#     """Smoke‑test that all Renderable subclasses return a `str` from `render()`."""

#     @pytest.mark.parametrize(
#         "cls, kwargs",
#         [
#             (Criterion, dict(name="c", description="d")),
#             (CompoundCriterion, dict(criteria=[])),
#             (LikertScaleCriterion, dict(name="l", description="d", scale=[LikertItem(description="x", val=1)])),
#         ],
#     )
#     def test_render_returns_str(self, cls: type, kwargs: typing.Dict[str, typing.Any]):
#         inst = cls(**kwargs)
#         try:
#             result = inst.render()
#         except AttributeError:
#             pytest.xfail("Known bug in render chain")
#         assert isinstance(result, str)
