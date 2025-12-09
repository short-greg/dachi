"""Updated behaviour‑tree unit tests.

These tests have been adapted to the new asynchronous task execution model and
`BaseModule` keyword‑initialisation.  **No tests were added or removed – every
original case remains, simply modernised.**

Google‑style docstrings and minimal comments are retained per project
conventions.  All async tests use `pytest.mark.asyncio`.
"""

import asyncio
import types
import random
import pytest
import pydantic
from dachi.core import Runtime, PrivateRuntime
from dachi.act.comm import Scope
from dachi.act.bt._core import TaskStatus
from dachi.act.bt._leafs import Action, Condition, WaitCondition, FixedTimer, RandomTimer, CountLimit

import time


def create_test_ctx():
    """Helper to create test context"""
    scope = Scope()
    return scope.ctx()

class ImmediateAction(Action):
    """A task that immediately returns a preset *status_val*."""

    status_val: TaskStatus = TaskStatus.SUCCESS

    async def execute(self) -> TaskStatus:
        return self.status_val


class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def execute(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: int = 4

    async def execute(self) -> TaskStatus:  # noqa: D401
        print('Acting!')
        return TaskStatus.FAILURE if self.value < 0 else TaskStatus.SUCCESS


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def execute(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: int = 4
    _count: Runtime[int] = PrivateRuntime(0)

    async def execute(self) -> TaskStatus:  # noqa: D401
        if self.value == 0:
            return TaskStatus.FAILURE
        self._count.set(self._count.get() + 1)
        if self._count.get() == 2:
            return TaskStatus.SUCCESS
        if self._count.get() < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


class _FlagWaitCond(WaitCondition):
    """WaitCondition whose outcome is controlled via the *flag* attribute."""

    flag: bool = True

    async def execute(self) -> bool:
        return self.flag


@pytest.mark.asyncio
class TestAction:
    async def test_storage_action_count_is_1(self):
        action = SetStorageAction(value=1)
        scope = Scope()
        ctx = scope.ctx()
        assert await action.tick(ctx) == TaskStatus.SUCCESS

    async def test_store_action_returns_fail_if_fail(self):
        action = SetStorageAction(value=-1)
        ctx = create_test_ctx()
        assert await action.tick(ctx) == TaskStatus.FAILURE

    async def test_running_after_one_tick(self):
        action = SetStorageActionCounter(value=1)
        ctx = create_test_ctx()
        assert await action.tick(ctx) == TaskStatus.RUNNING

    async def test_success_after_two_tick(self):
        action = SetStorageActionCounter(value=2)
        ctx = create_test_ctx()
        await action.tick(ctx)
        assert await action.tick(ctx) == TaskStatus.SUCCESS

    async def test_ready_after_reset(self):
        action = SetStorageActionCounter(value=2)
        ctx = create_test_ctx()
        await action.tick(ctx)
        action.reset()
        assert action.status == TaskStatus.READY

@pytest.mark.asyncio
class TestTimers:
    async def test_fixed_timer(self, monkeypatch):
        start = 0.0; monkeypatch.setattr(time, "time", lambda: start)
        timer = FixedTimer(seconds=5)
        assert await timer.tick(create_test_ctx()) is TaskStatus.RUNNING
        monkeypatch.setattr(time, "time", lambda: start + 6)
        assert await timer.tick(create_test_ctx()) is TaskStatus.SUCCESS

    async def test_random_timer(self, monkeypatch):
        monkeypatch.setattr(random, "random", lambda: 0.5)  # deterministic
        t0 = 100.0; monkeypatch.setattr(
            time, "time", lambda: t0
        )
        rt = RandomTimer(seconds_lower=2, seconds_upper=4)
        assert await rt.tick(create_test_ctx()) is TaskStatus.RUNNING
        monkeypatch.setattr(time, "time", lambda: t0 + 3.5)
        assert await rt.tick(create_test_ctx()) is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCondition:
    async def test_condition_returns_success(self):
        condition = SampleCondition(x=1)
        assert await condition.tick(create_test_ctx()) == TaskStatus.SUCCESS

    async def test_condition_returns_failure(self):
        condition = SampleCondition(x=-1)
        assert await condition.tick(create_test_ctx()) == TaskStatus.FAILURE

    async def test_condition_status_is_success(self):
        condition = SampleCondition(x=-1)
        await condition.tick(create_test_ctx())
        assert condition.status == TaskStatus.FAILURE

    async def test_condition_status_is_ready_after_reset(self):
        condition = SampleCondition(x=-1)
        await condition.tick(create_test_ctx())
        condition.reset()
        assert condition.status == TaskStatus.READY


@pytest.mark.asyncio
class TestWaitCondition:
    """`WaitCondition` maps *False* → WAITING and *True* → SUCCESS."""

    async def test_waiting_when_false(self):
        cond = _FlagWaitCond(flag=False)
        assert await cond.tick(create_test_ctx()) is TaskStatus.WAITING
        assert cond.status is TaskStatus.WAITING

    async def test_success_when_true(self):
        cond = _FlagWaitCond(flag=True)
        assert await cond.tick(create_test_ctx()) is TaskStatus.SUCCESS
        assert cond.status is TaskStatus.SUCCESS

    async def test_reset_restores_ready(self):
        cond = _FlagWaitCond(flag=False)
        await cond.tick(create_test_ctx())
        cond.reset()
        assert cond.status is TaskStatus.READY


class ToggleWait(WaitCondition):
    """Returns WAITING on first tick, SUCCESS on second."""

    _first: bool = pydantic.PrivateAttr(default=True)

    async def execute(self):
        if self._first:
            self._first = False
            return False
        return True


@pytest.mark.asyncio
class TestToggleWaitCondition:
    async def test_wait_condition_wait_then_success(self):
        cond = ToggleWait()
        assert await cond.tick(create_test_ctx()) is TaskStatus.WAITING
        assert await cond.tick(create_test_ctx()) is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCountLimit:
    async def test_runs_until_count_then_success(self):
        cl = CountLimit(count=3)
        assert await cl.tick(create_test_ctx()) is TaskStatus.RUNNING
        assert await cl.tick(create_test_ctx()) is TaskStatus.RUNNING
        assert await cl.tick(create_test_ctx()) is TaskStatus.SUCCESS

    async def test_countlimit_reset(self):
        cl = CountLimit(count=2)
        await cl.tick(create_test_ctx()); await cl.tick(create_test_ctx())
        cl.reset()
        assert cl.status is TaskStatus.READY
        assert await cl.tick(create_test_ctx()) is TaskStatus.RUNNING


class TestPortSystem:
    """Test the port declaration and processing system for leaf classes"""
    
    def test_process_ports_extracts_annotations_from_simple_class(self):
        from dachi.act.bt._core import LeafTask
        
        class TestInputs:
            param1: int
            param2: str
        
        result = LeafTask._process_ports(TestInputs)
        
        assert "param1" in result
        assert result["param1"]["type"] == int
        assert "param2" in result  
        assert result["param2"]["type"] == str
    
    def test_process_ports_extracts_defaults_from_class(self):
        from dachi.act.bt._core import LeafTask
        
        class TestInputs:
            param1: int = 5
            param2: str = "default"
            param3: float
        
        result = LeafTask._process_ports(TestInputs)
        
        assert result["param1"]["default"] == 5
        assert result["param2"]["default"] == "default"
        assert "default" not in result["param3"]  # No default provided
    
    def test_action_with_inputs_creates_ports(self):
        class TestAction(Action):
            class inputs:
                param1: int
                param2: str = "default"
        
        assert hasattr(TestAction, '__ports__')
        assert "inputs" in TestAction.__ports__
        assert "param1" in TestAction.__ports__["inputs"]
        assert TestAction.__ports__["inputs"]["param1"]["type"] == int
        assert TestAction.__ports__["inputs"]["param2"]["default"] == "default"
    
    def test_action_with_outputs_creates_ports(self):
        from typing import TypedDict
        
        class TestAction(Action):
            class outputs(TypedDict):
                result: int
                success: bool
        
        assert hasattr(TestAction, '__ports__')
        assert "outputs" in TestAction.__ports__
        assert "result" in TestAction.__ports__["outputs"]
        assert TestAction.__ports__["outputs"]["result"]["type"] == int
        assert TestAction.__ports__["outputs"]["success"]["type"] == bool
    
    def test_action_without_ports_has_empty_ports(self):
        class TestAction(Action):
            pass
        
        assert hasattr(TestAction, '__ports__')
        assert TestAction.__ports__["inputs"] == {}
        assert TestAction.__ports__["outputs"] == {}
    
    def test_condition_with_ports_works(self):
        class TestCondition(Condition):
            class inputs:
                threshold: float = 0.5
        
        ports = TestCondition.__ports__
        assert "threshold" in ports["inputs"]
        assert ports["inputs"]["threshold"]["default"] == 0.5
    
    def test_inheritance_preserves_port_processing(self):
        class BaseAction(Action):
            class inputs:
                base_param: int

        class DerivedAction(BaseAction):
            class inputs:
                derived_param: str

        # Each class should have its own ports
        assert "base_param" in BaseAction.__ports__["inputs"]
        assert "base_param" not in DerivedAction.__ports__["inputs"]
        assert "derived_param" in DerivedAction.__ports__["inputs"]


class TestLeafSchemaMetadata:
    """Test the schema method includes port metadata for Leaf classes"""

    def test_schema_returns_dict_with_xports_key(self):
        class TestAction(Action):
            class inputs:
                param1: int

            class outputs:
                result: str

        schema = TestAction.schema()

        assert isinstance(schema, dict)
        assert "x-ports" in schema
        assert "properties" in schema

    def test_schema_includes_inputs_metadata(self):
        class TestAction(Action):
            class inputs:
                param1: int
                param2: str = "default"

            async def execute(self, param1, param2):
                return TaskStatus.SUCCESS

        schema = TestAction.schema()

        assert "x-ports" in schema
        assert "inputs" in schema["x-ports"]
        assert "param1" in schema["x-ports"]["inputs"]
        assert "param2" in schema["x-ports"]["inputs"]
        assert schema["x-ports"]["inputs"]["param1"]["type"] == "integer"
        assert schema["x-ports"]["inputs"]["param2"]["type"] == "string"
        assert schema["x-ports"]["inputs"]["param2"]["default"] == "default"

    def test_schema_includes_outputs_metadata(self):
        class TestAction(Action):
            class outputs:
                result: int
                success: bool

            async def execute(self):
                return TaskStatus.SUCCESS

        schema = TestAction.schema()

        assert "x-ports" in schema
        assert "outputs" in schema["x-ports"]
        assert "result" in schema["x-ports"]["outputs"]
        assert "success" in schema["x-ports"]["outputs"]
        assert schema["x-ports"]["outputs"]["result"]["type"] == "integer"
        assert schema["x-ports"]["outputs"]["success"]["type"] == "boolean"

    def test_schema_with_no_ports_has_no_xports_key(self):
        class TestAction(Action):
            async def execute(self):
                return TaskStatus.SUCCESS

        schema = TestAction.schema()

        assert "x-ports" not in schema

    def test_schema_base_schema_is_preserved(self):
        class TestAction(Action):
            param: int = 5

            class inputs:
                input1: str

        schema = TestAction.schema()

        assert "properties" in schema
        assert "param" in schema["properties"]

    def test_condition_schema_includes_ports(self):
        class TestCondition(Condition):
            class inputs:
                threshold: float = 0.5

            async def execute(self, threshold):
                return True

        schema = TestCondition.schema()

        assert "x-ports" in schema
        assert "inputs" in schema["x-ports"]
        assert "threshold" in schema["x-ports"]["inputs"]
        assert schema["x-ports"]["inputs"]["threshold"]["type"] == "number"
        assert schema["x-ports"]["inputs"]["threshold"]["default"] == 0.5
