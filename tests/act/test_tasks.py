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
from dachi.act import _tasks as behavior
from dachi.act._core import TaskStatus

from dachi.act import _tasks as behavior
from dachi.act._core import TaskStatus
from dachi.core import ModuleList, Attr


class ATask(behavior.Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def act(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(behavior.Action):
    """Action whose success/failure depends on *value*."""

    value: InitVar[int] = 4

    def __post_init__(self, value: int):
        # TODO: enforce post init is called
        super().__post_init__()
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        print('Acting!')
        return TaskStatus.FAILURE if self.value.data < 0 else TaskStatus.SUCCESS


class SampleCondition(behavior.Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def condition(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(behavior.Action):
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
class TestSequence:
    async def test_sequence_is_running_when_started(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        assert await sequence.tick() == TaskStatus.RUNNING

    async def test_sequence_is_success_when_finished(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = behavior.Sequence(tasks=[action1, action2])
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.SUCCESS

    async def test_sequence_is_failure_less_than_zero(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = behavior.Sequence(tasks=[action1, action2])
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.FAILURE

    async def test_sequence_is_ready_when_reset(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = behavior.Sequence(tasks=[action1, action2])
        await sequence.tick()
        await sequence.tick()
        sequence.reset()
        assert sequence.status == TaskStatus.READY

    async def test_sequence_finished_after_three_ticks(self):
        action1 = SetStorageAction(value=2)
        action2 = SetStorageActionCounter(value=3)
        sequence = behavior.Sequence(tasks=[action1, action2])
        await sequence.tick()
        await sequence.tick()
        assert await sequence.tick() == TaskStatus.SUCCESS


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
class TestFallback:
    async def test_fallback_is_successful_after_one_tick(self):
        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction()
        fallback = behavior.Selector(tasks=[action1, action2])
        assert await fallback.tick() == TaskStatus.SUCCESS

    async def test_fallback_is_successful_after_two_ticks(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=1)
        fallback = behavior.Selector(tasks=[action1, action2])
        await fallback.tick()
        assert await fallback.tick() == TaskStatus.SUCCESS

    async def test_fallback_fails_after_two_ticks(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(tasks=[action1, action2])
        await fallback.tick()
        assert await fallback.tick() == TaskStatus.FAILURE

    async def test_fallback_running_after_one_tick(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(tasks=[action1, action2])
        assert await fallback.tick() == TaskStatus.RUNNING

    async def test_fallback_ready_after_reset(self):
        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(tasks=[action1, action2])
        await fallback.tick()
        fallback.reset()
        assert fallback.status == TaskStatus.READY


@pytest.mark.asyncio
class TestAsLongAs:
    async def test_while_fails_if_failure(self):
        action1 = SetStorageActionCounter(value=0)
        action1._count = -1
        while_ = behavior.AsLongAs(task=action1, target_status=TaskStatus.FAILURE)
        await while_.tick()
        action1.value = 1
        assert while_.status == TaskStatus.RUNNING

    async def test_aslongas_fails_if_failure_after_two(self):
        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        action1.value.data = 4
        aslongas = behavior.AsLongAs(task=action1)
        await aslongas.tick()
        action1.value.data = 0
        assert await aslongas.tick() == TaskStatus.FAILURE


@pytest.mark.asyncio
class TestUntil:
    async def test_until_successful_if_success(self):
        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        until_ = behavior.Until(task=action1)
        assert await until_.tick() == TaskStatus.SUCCESS

    async def test_until_successful_if_success_after_two(self):
        action1 = SetStorageActionCounter(value=0)
        action1._count = 1
        until_ = behavior.Until(task=action1)
        await until_.tick()
        action1.value.data = 1
        assert await until_.tick() == TaskStatus.SUCCESS


# ---------------------------------------------------------------------------
# Helper stubs used exclusively by this file
# ---------------------------------------------------------------------------
class _ImmediateAction(behavior.Action):
    """A task that immediately returns a preset *status_val*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._ret = status_val

    async def act(self) -> TaskStatus:
        return self._ret


class _FlagWaitCond(behavior.WaitCondition):
    """WaitCondition whose outcome is controlled via the *flag* attribute."""

    flag: bool = True

    async def condition(self) -> bool:
        return self.flag


# ---------------------------------------------------------------------------
# Tests start here – each class targets a single public surface.
# ---------------------------------------------------------------------------
class TestParallelValidate:
    """`Parallel.validate` should enforce quorum invariants and raise early."""

    def _parallel(self, fails: int, succ: int, n: int = 3):
        tasks = ModuleList(data=[_ImmediateAction(status_val=TaskStatus.RUNNING) for _ in range(n)])
        return behavior.Parallel(tasks=tasks, fails_on=fails, succeeds_on=succ)

    def test_ok_when_thresholds_within_bounds(self):
        par = self._parallel(fails=1, succ=-1)  # always succeed with *all* successes
        # Should not raise
        par.validate()

    def test_error_when_threshold_exceeds_task_count(self):
        par = self._parallel(fails=2, succ=3, n=3)
        # 2 + 2 - 1 > 3 triggers the guard
        with pytest.raises(ValueError):
            par.validate()

    def test_error_when_zero_threshold(self):
        par = self._parallel(fails=0, succ=1)
        with pytest.raises(ValueError):
            par.validate()


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


class TestStateFuncDecorator:
    """`statefunc` must simply flag the target callable with *_is_state*."""

    def test_flag_added(self):
        @behavior.statefunc
        def dummy():
            pass

        assert getattr(dummy, "_is_state", False) is True



# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

class ImmediateAction(behavior.Action):
    """A task that immediately returns a fixed *status*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._status_val = status_val

    async def act(self) -> TaskStatus:  # noqa: D401
        return self._status_val


@pytest.mark.asyncio
class TestRoot:
    async def test_root_without_child(self):
        assert await behavior.BT().tick() is TaskStatus.SUCCESS

    async def test_root_delegates(self):
        r = behavior.BT(root=ImmediateAction(status_val=TaskStatus.FAILURE))
        assert await r.tick() is TaskStatus.FAILURE

    async def test_root_reset_propagates(self):
        child = ImmediateAction(status_val=TaskStatus.SUCCESS)
        r = behavior.BT(root=child)
        await r.tick(); r.reset()
        assert child.status is TaskStatus.READY


class TestSerialValidation:
    def test_list_to_modulelist(self):
        serial = behavior.Serial(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)])
        assert isinstance(serial.tasks, ModuleList)

    def test_defaults_to_empty(self):
        assert len(behavior.Serial().tasks) == 0

    def test_invalid_tasks_type(self):
        with pytest.raises(ValueError):
            behavior.Serial(tasks=123)


@pytest.mark.asyncio
class TestParallel:
    async def test_all_success(self):
        par = behavior.Parallel(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS) for _ in range(3)], succeeds_on=-1, fails_on=1)
        assert await par.tick() is TaskStatus.SUCCESS

    async def test_failure_threshold(self):
        par = behavior.Parallel(tasks=[ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)], fails_on=1, succeeds_on=2)
        assert await par.tick() is TaskStatus.RUNNING

    async def test_running_until_quorum(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)]
        par = behavior.Parallel(tasks=tasks, fails_on=2, succeeds_on=2)
        assert await par.tick() is TaskStatus.RUNNING

    async def test_fails_on_1_failure(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE)]
        par = behavior.Parallel(tasks=tasks, fails_on=1, succeeds_on=2)
        assert await par.tick() is TaskStatus.FAILURE

#     # async def test_reset_propagates(self):
#     #     inner = SequenceAction([TaskStatus.RUNNING, TaskStatus.SUCCESS])
#     #     par = behavior.Parallel(tasks=[inner], fails_on=1, succeeds_on=1)
#     #     await par.tick(); par.reset()
#     #     assert inner.status is TaskStatus.READY


@pytest.mark.asyncio
class TestNotDecorator:
    async def test_invert_success(self):
        assert await behavior.Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS)).tick() is TaskStatus.FAILURE

    async def test_invert_failure(self):
        assert await behavior.Not(task=ImmediateAction(status_val=TaskStatus.FAILURE)).tick() is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestTimers:
    async def test_fixed_timer(self, monkeypatch):
        start = 0.0; monkeypatch.setattr(behavior.time, "time", lambda: start)
        timer = behavior.FixedTimer(seconds=5)
        assert await timer.tick() is TaskStatus.RUNNING
        monkeypatch.setattr(behavior.time, "time", lambda: start + 6)
        assert await timer.tick() is TaskStatus.SUCCESS

    async def test_random_timer(self, monkeypatch):
        monkeypatch.setattr(behavior.random, "random", lambda: 0.5)  # deterministic
        t0 = 100.0; monkeypatch.setattr(
            behavior.time, "time", lambda: t0
        )
        rt = behavior.RandomTimer(seconds_lower=2, seconds_upper=4)
        assert await rt.tick() is TaskStatus.RUNNING
        monkeypatch.setattr(behavior.time, "time", lambda: t0 + 3.5)
        assert await rt.tick() is TaskStatus.SUCCESS



# # ---------------------------------------------------------------------------
# # 14. Loop context‑manager utilities ----------------------------------------
# # ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLoopUtilities:
    async def test_loop_aslongas_async_context_invalid(self):
        cm = behavior.loop_aslongas(task=ImmediateAction(status_val=TaskStatus.SUCCESS), status=TaskStatus.SUCCESS)
        with pytest.raises(TypeError):
            async with cm:  # function is not a valid async context‑manager
                pass

    async def test_loop_until_async_context_invalid(self):
        cm = behavior.loop_until(task=ImmediateAction(status_val=TaskStatus.SUCCESS))
        with pytest.raises(TypeError):
            async with cm:
                pass

# # ---------------------------------------------------------------------------
# # 15. WaitCondition ----------------------------------------------------------
# # ---------------------------------------------------------------------------

class ToggleWait(behavior.WaitCondition):
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

# # ---------------------------------------------------------------------------
# # 16. CountLimit -------------------------------------------------------------
# # ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCountLimit:
    async def test_runs_until_count_then_success(self):
        cl = behavior.CountLimit(count=3)
        assert await cl.tick() is TaskStatus.RUNNING
        assert await cl.tick() is TaskStatus.RUNNING
        assert await cl.tick() is TaskStatus.SUCCESS

    async def test_countlimit_reset(self):
        cl = behavior.CountLimit(count=2)
        await cl.tick(); await cl.tick()
        cl.reset()
        assert cl.status is TaskStatus.READY
        assert await cl.tick() is TaskStatus.RUNNING

# # ---------------------------------------------------------------------------
# # 17. PreemptCond ------------------------------------------------------------
# # ---------------------------------------------------------------------------

class AlwaysTrueCond(behavior.Condition):
    async def condition(self):
        return True

class AlwaysFalseCond(behavior.Condition):
    async def condition(self):
        return False

@pytest.mark.asyncio
class TestPreemptCond:
    async def test_preemptcond_failure_when_false(self):
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = behavior.PreemptCond(conds=[AlwaysFalseCond()], task=main)
        assert await pc.tick() is TaskStatus.FAILURE
        assert main.status is TaskStatus.READY  # main skipped

    async def test_preemptcond_propagates_task_success(self):
        main = ImmediateAction(status_val=TaskStatus.SUCCESS)
        pc = behavior.PreemptCond(conds=[AlwaysTrueCond()], task=main)
        assert await pc.tick() is TaskStatus.SUCCESS



# @pytest.mark.asyncio
# class TestRunTask:
#     async def test_run_task_yields(self):
#         act = SequenceAction([TaskStatus.RUNNING, TaskStatus.SUCCESS])
#         collected = [s async for s in behavior.run_task(act, interval=None)]
#         assert collected == [TaskStatus.RUNNING, TaskStatus.SUCCESS]
#         assert act.status is TaskStatus.SUCCESS

