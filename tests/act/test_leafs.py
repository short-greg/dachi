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
from dachi.core import InitVar, Attr
from dachi.act._bt._core import TaskStatus
from dachi.act._bt._leafs import Action, Condition, WaitCondition, FixedTimer, RandomTimer, CountLimit

import time

class ImmediateAction(Action):
    """A task that immediately returns a preset *status_val*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._ret = status_val

    async def act(self) -> TaskStatus:
        return self._ret


class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def act(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: InitVar[int] = 4

    def __post_init__(self, value: int):
        # TODO: enforce post init is called
        super().__post_init__()
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        print('Acting!')
        return TaskStatus.FAILURE if self.value.data < 0 else TaskStatus.SUCCESS


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def condition(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: InitVar[int] = 4

    def __post_init__(self, value: int=4):
        super().__post_init__()
        self._count = 0
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        if self.value.data == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING



class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def act(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: InitVar[int] = 4

    def __post_init__(self, value: int):
        # TODO: enforce post init is called
        super().__post_init__()
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        print('Acting!')
        return TaskStatus.FAILURE if self.value.data < 0 else TaskStatus.SUCCESS


class _FlagWaitCond(WaitCondition):
    """WaitCondition whose outcome is controlled via the *flag* attribute."""

    flag: bool = True

    async def condition(self) -> bool:
        return self.flag


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def condition(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: InitVar[int] = 4

    def __post_init__(self, value: int=4):
        super().__post_init__()
        self._count = 0
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        if self.value.data == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


@pytest.mark.asyncio
class TestAction:
    async def test_storage_action_count_is_1(self):
        action = SetStorageAction(value=1)
        assert await action.tick() == TaskStatus.SUCCESS

    async def test_store_action_returns_fail_if_fail(self):
        action = SetStorageAction(value=-1)
        assert await action.tick() == TaskStatus.FAILURE

    async def test_running_after_one_tick(self):
        action = SetStorageActionCounter(value=1)
        assert await action.tick() == TaskStatus.RUNNING

    async def test_success_after_two_tick(self):
        action = SetStorageActionCounter(value=2)
        await action.tick()
        assert await action.tick() == TaskStatus.SUCCESS

    async def test_ready_after_reset(self):
        action = SetStorageActionCounter(value=2)
        await action.tick()
        action.reset()
        assert action.status == TaskStatus.READY

    async def test_load_state_dict_sets_state(self):
        action = SetStorageActionCounter(value=3)
        action2 = SetStorageActionCounter(value=2)
        await action.tick()
        action2.load_state_dict(action.state_dict())
        assert action2.value.data == 3



@pytest.mark.asyncio
class TestTimers:
    async def test_fixed_timer(self, monkeypatch):
        start = 0.0; monkeypatch.setattr(time, "time", lambda: start)
        timer = FixedTimer(seconds=5)
        assert await timer.tick() is TaskStatus.RUNNING
        monkeypatch.setattr(time, "time", lambda: start + 6)
        assert await timer.tick() is TaskStatus.SUCCESS

    async def test_random_timer(self, monkeypatch):
        monkeypatch.setattr(random, "random", lambda: 0.5)  # deterministic
        t0 = 100.0; monkeypatch.setattr(
            time, "time", lambda: t0
        )
        rt = RandomTimer(seconds_lower=2, seconds_upper=4)
        assert await rt.tick() is TaskStatus.RUNNING
        monkeypatch.setattr(time, "time", lambda: t0 + 3.5)
        assert await rt.tick() is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestCondition:
    async def test_condition_returns_success(self):
        condition = SampleCondition(x=1)
        assert await condition.tick() == TaskStatus.SUCCESS

    async def test_condition_returns_failure(self):
        condition = SampleCondition(x=-1)
        assert await condition.tick() == TaskStatus.FAILURE

    async def test_condition_status_is_success(self):
        condition = SampleCondition(x=-1)
        await condition.tick()
        assert condition.status == TaskStatus.FAILURE

    async def test_condition_status_is_ready_after_reset(self):
        condition = SampleCondition(x=-1)
        await condition.tick()
        condition.reset()
        assert condition.status == TaskStatus.READY


@pytest.mark.asyncio
class TestWaitCondition:
    """`WaitCondition` maps *False* → WAITING and *True* → SUCCESS."""

    async def test_waiting_when_false(self):
        cond = _FlagWaitCond(flag=False)
        assert await cond.tick() is TaskStatus.WAITING
        assert cond.status is TaskStatus.WAITING

    async def test_success_when_true(self):
        cond = _FlagWaitCond(flag=True)
        assert await cond.tick() is TaskStatus.SUCCESS
        assert cond.status is TaskStatus.SUCCESS

    async def test_reset_restores_ready(self):
        cond = _FlagWaitCond(flag=False)
        await cond.tick()
        cond.reset()
        assert cond.status is TaskStatus.READY



@pytest.mark.asyncio
class TestCountLimit:
    async def test_runs_until_count_then_success(self):
        cl = CountLimit(count=3)
        assert await cl.tick() is TaskStatus.RUNNING
        assert await cl.tick() is TaskStatus.RUNNING
        assert await cl.tick() is TaskStatus.SUCCESS

    async def test_countlimit_reset(self):
        cl = CountLimit(count=2)
        await cl.tick(); await cl.tick()
        cl.reset()
        assert cl.status is TaskStatus.READY
        assert await cl.tick() is TaskStatus.RUNNING


class ToggleWait(WaitCondition):
    """Returns WAITING on first tick, SUCCESS on second."""

    def __init__(self):
        super().__init__()
        self._first = True

    async def condition(self):
        if self._first:
            self._first = False
            return False
        return True

@pytest.mark.asyncio
class TestWaitCondition:
    async def test_wait_condition_wait_then_success(self):
        cond = ToggleWait()
        assert await cond.tick() is TaskStatus.WAITING
        assert await cond.tick() is TaskStatus.SUCCESS


class TestPortSystem:
    """Test the port declaration and processing system for leaf classes"""
    
    def test_process_ports_extracts_annotations_from_simple_class(self):
        from dachi.act._bt._core import Leaf
        
        class TestInputs:
            param1: int
            param2: str
        
        result = Leaf._process_ports(TestInputs)
        
        assert "param1" in result
        assert result["param1"]["type"] == int
        assert "param2" in result  
        assert result["param2"]["type"] == str
    
    def test_process_ports_extracts_defaults_from_class(self):
        from dachi.act._bt._core import Leaf
        
        class TestInputs:
            param1: int = 5
            param2: str = "default"
            param3: float
        
        result = Leaf._process_ports(TestInputs)
        
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
