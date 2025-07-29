from dachi.act import _core


class TestTaskStatus(object):

    def test_task_status_is_done_if_failed(self):

        assert _core.TaskStatus.FAILURE.is_done

    def test_task_status_is_done_if_success(self):

        assert _core.TaskStatus.SUCCESS.is_done

    def test_task_status_is_not_done_if_running(self):

        assert not _core.TaskStatus.RUNNING.is_done

    def test_task_status_in_progress_if_running(self):

        assert _core.TaskStatus.RUNNING.in_progress

    def test_task_status_success_if_SUCCESS(self):

        assert _core.TaskStatus.SUCCESS.success

    def test_task_status_not_success_if_FAILURE(self):

        assert not _core.TaskStatus.FAILURE.success

    def test_or_returns_success_if_one_success(self):

        assert (_core.TaskStatus.SUCCESS | _core.TaskStatus.FAILURE).success

    def test_or_returns_success_if_one_success_and_running(self):

        assert (_core.TaskStatus.SUCCESS | _core.TaskStatus.RUNNING).success

    def test_or_returns_running_if_failure_and_running(self):

        assert (_core.TaskStatus.FAILURE | _core.TaskStatus.RUNNING).running

    def test_and_returns_success_if_one_failure(self):

        assert (_core.TaskStatus.SUCCESS & _core.TaskStatus.FAILURE).failure

    def test_or_returns_success_if_one_success_and_running(self):

        assert (_core.TaskStatus.FAILURE & _core.TaskStatus.RUNNING).failure

    def test_or_returns_running_if_failure_and_running(self):

        assert (_core.TaskStatus.SUCCESS & _core.TaskStatus.RUNNING).running

    def test_invert_converts_failure_to_success(self):

        assert (_core.TaskStatus.FAILURE.invert()).success

    def test_invert_converts_success_to_failure(self):

        assert (_core.TaskStatus.SUCCESS.invert()).failure


class TestFromBool(object):

    def test_from_bool_returns_success_for_true(self):
        assert _core.from_bool(True) == _core.TaskStatus.SUCCESS

    def test_from_bool_returns_failure_for_false(self):
        assert _core.from_bool(False) == _core.TaskStatus.FAILURE


"""Additional black-box tests for ``dachi.act._core``.

Each class targets a single public surface area that is *not* fully covered by
``tests/act/test_core.py``.  Tests are nit-picky and check **one** observable
behaviour apiece (except when a function inherently yields multiple results
that must be asserted together).
"""

import asyncio
import types
import pytest

from dachi.act import _core as core

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

class _DummyTask(core.Task):
    """Concrete :class:`~core.Task` that simply returns a preset status."""

    def __init__(self, ret_status: core.TaskStatus):  # noqa: D401 – simple stub
        super().__init__()
        self._ret_status = ret_status

    async def tick(self) -> core.TaskStatus:  # noqa: D401 – trivial impl
        self._status.set(self._ret_status)
        return self._ret_status

    # sync variant used in rare cases where a synchronous path is required
    def sync_tick(self) -> core.TaskStatus:  # noqa: D401
        self._status.set(self._ret_status)
        return self._ret_status


class _CountingFTask(core.FTask):
    """FTask stub that counts ``func_tick`` invocations for idempotency tests."""

    def __init__(self, ret_status: core.TaskStatus = core.TaskStatus.SUCCESS):
        super().__init__(name="ft", args=[], kwargs={})
        self._ret_status = ret_status
        self.calls: int = 0
        self.obj = object()  # satisfy FTask pre-condition

    async def func_tick(self) -> core.TaskStatus:  # noqa: D401 – simple stub
        self.calls += 1
        return self._ret_status


class TestTaskStatusProperties:
    """Flags reflect the semantic meaning of each status."""

    def test_is_done_success_failure(self):
        assert core.TaskStatus.SUCCESS.is_done is True
        assert core.TaskStatus.FAILURE.is_done is True
        assert core.TaskStatus.RUNNING.is_done is False

    def test_in_progress_running_waiting(self):
        assert core.TaskStatus.RUNNING.in_progress is True
        assert core.TaskStatus.WAITING.in_progress is True
        assert core.TaskStatus.READY.in_progress is False

    def test_ready_flag(self):
        assert core.TaskStatus.READY.ready is True
        assert core.TaskStatus.SUCCESS.ready is False

    def test_failure_success_running_flags(self):
        assert core.TaskStatus.FAILURE.failure is True
        assert core.TaskStatus.SUCCESS.success is True
        assert core.TaskStatus.RUNNING.running is True


class TestFromBoolHelper:
    """:pyfunc:`~core.from_bool` mirrors :pyfunc:`TaskStatus.from_bool`."""

    @pytest.mark.parametrize("value, expected", [(True, core.TaskStatus.SUCCESS), (False, core.TaskStatus.FAILURE)])
    def test_from_bool(self, value, expected):
        assert core.from_bool(value) is expected


class TestTaskStatusOperators:
    """``__or__`` and ``__and__`` follow documented precedence rules."""

    # ----- OR -----
    def test_or_success_dominates(self):
        assert (core.TaskStatus.SUCCESS | core.TaskStatus.RUNNING) is core.TaskStatus.SUCCESS

    def test_or_running_over_waiting(self):
        assert (core.TaskStatus.RUNNING | core.TaskStatus.WAITING) is core.TaskStatus.RUNNING

    def test_or_waiting_over_ready(self):
        assert (core.TaskStatus.WAITING | core.TaskStatus.READY) is core.TaskStatus.WAITING

    # ----- AND -----
    def test_and_failure_dominates(self):
        assert (core.TaskStatus.FAILURE & core.TaskStatus.SUCCESS) is core.TaskStatus.FAILURE

    def test_and_running_over_success(self):
        assert (core.TaskStatus.RUNNING & core.TaskStatus.SUCCESS) is core.TaskStatus.RUNNING

    def test_and_waiting_over_ready(self):
        assert (core.TaskStatus.WAITING & core.TaskStatus.READY) is core.TaskStatus.WAITING

    # ----- invert -----
    def test_invert_roundtrip(self):
        assert core.TaskStatus.SUCCESS.invert() is core.TaskStatus.FAILURE
        assert core.TaskStatus.FAILURE.invert() is core.TaskStatus.SUCCESS
        assert core.TaskStatus.RUNNING.invert() is core.TaskStatus.RUNNING  # unchanged


# ---------------------------------------------------------------------------
# Task – life-cycle behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTaskLifecycle:
    """A *Task* starts READY, updates on *tick*, and can be *reset*."""

    async def test_initial_status_ready(self):
        t = _DummyTask(core.TaskStatus.SUCCESS)
        assert t.status is core.TaskStatus.READY

    async def test_tick_updates_status_and_returns(self):
        t = _DummyTask(core.TaskStatus.SUCCESS)
        out = await t.tick()
        assert out is core.TaskStatus.SUCCESS
        assert t.status is core.TaskStatus.SUCCESS

    async def test_call_delegates_to_tick(self):
        t = _DummyTask(core.TaskStatus.FAILURE)
        out = await t()
        assert out is core.TaskStatus.FAILURE

    async def test_reset_restores_ready(self):
        t = _DummyTask(core.TaskStatus.SUCCESS)
        await t.tick()
        t.reset()
        assert t.status is core.TaskStatus.READY


# # ---------------------------------------------------------------------------
# # FTask – object pre-condition, idempotency, and reset
# # ---------------------------------------------------------------------------

# @pytest.mark.asyncio
# class TestFTaskBehaviour:
#     """*FTask* enforces `obj` presence and caches *done* state."""

#     async def test_tick_without_obj_raises(self):
#         task = core.FTask(name="ft", args=[], kwargs={})
#         with pytest.raises(ValueError):
#             await task.tick()

#     async def test_tick_executes_and_caches_result(self):
#         task = _CountingFTask()
#         out = await task.tick()
#         assert out is core.TaskStatus.SUCCESS
#         assert task.calls == 1
#         # second call – should short-circuit and *not* call func_tick again
#         out2 = await task.tick()
#         assert out2 is core.TaskStatus.SUCCESS
#         assert task.calls == 1

#     async def test_reset_clears_done_state(self):
#         task = _CountingFTask()
#         await task.tick()
#         task.reset()
#         assert task.status is core.TaskStatus.READY
#         # after reset, func_tick should execute again
#         await task.tick()
#         assert task.calls == 2

