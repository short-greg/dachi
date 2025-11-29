# # tests/act/test_states.py
# import pytest
# from typing import Any
# import typing as t

# from dachi.act import _states as S
# from dachi.act._bt._core import Task, TaskStatus
# from dachi.act._bt._leafs import Action
# from dachi.proc import Process, AsyncProcess

# from .utils import (
#     _SyncBoolProc, _AsyncBoolProc, 
#     _ImmediateTask, _ToggleTask, 
#     _ABoolProc, _BoolProc
# )


# @pytest.mark.asyncio
# class TestBranchState:
#     """`BranchState.update` maps (a) truthiness to TaskStatus and
#     (b) propagates exceptions unchanged."""

#     # ---- truthiness mapping (sync) ----
#     @pytest.mark.parametrize("value, expected", [
#         (True,  TaskStatus.SUCCESS),
#         (1,     TaskStatus.SUCCESS),      # non-bool truthy
#         (False, TaskStatus.FAILURE),
#         (0,     TaskStatus.FAILURE),      # non-bool falsy
#     ])
#     async def test_sync_process(self, value: Any, expected: TaskStatus) -> None:
#         st = S.BranchState(f=_SyncBoolProc(value))
#         assert await st.update() is expected

#     @pytest.mark.parametrize("value, expected", [
#         (True,  TaskStatus.SUCCESS),
#         (False, TaskStatus.FAILURE),
#     ])
#     async def test_async_process(self, value: bool, expected: TaskStatus) -> None:
#         st = S.BranchState(f=_AsyncBoolProc(val=value))
#         assert await st.update() is expected

#     # ---- exception bubbling ----
#     async def test_exception_propagation(self) -> None:
#         class _Boom(Process):
#             def forward(self):  # noqa: D401
#                 raise RuntimeError("boom")

#         with pytest.raises(RuntimeError):
#             await S.BranchState(f=_Boom()).update()

# @pytest.mark.asyncio
# class TestTaskStatePassThrough:
#     """`TaskState.update` must return *exactly* what the wrapped task returns."""

#     @pytest.mark.parametrize("status", list(TaskStatus))
#     async def test_pass_through(self, status: TaskStatus) -> None:
#         ts = S.TaskState(task=_ImmediateTask(status))
#         assert await ts.update() is status


# @pytest.mark.asyncio
# class TestTaskStateBehaviour:
#     """Extra behavioural guarantees for TaskState."""

#     async def test_tick_called_once(self) -> None:
#         counter = {"n": 0}

#         class _CountingTask(Task):
#             async def tick(self_inner):  # noqa: D401
#                 counter["n"] += 1
#                 return TaskStatus.SUCCESS

#         ts = S.TaskState(task=_CountingTask())
#         await ts.update()
#         assert counter["n"] == 1

#     async def test_exception_propagation(self) -> None:
#         class _Boom(Task):
#             async def tick(self):  # noqa: D401
#                 raise ValueError("kaboom")

#         with pytest.raises(ValueError):
#             await S.TaskState(task=_Boom()).update()


# def _state(fn):
#     fn._is_state = True      # noqa: SLF001
#     return fn



# # ---------------------------------------------------------------------------
# # BranchState
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# class TestBranchStateMore:
#     """Edge-cases for truthiness handling and type errors."""

#     @pytest.mark.parametrize("val", [1, True])
#     async def test_async_non_bool_truthy(self, val):
#         st = S.BranchState(f=_ABoolProc(val))
#         assert await st.update() is TaskStatus.SUCCESS

#     async def test_sync_none_is_failure(self):
#         st = S.BranchState(f=_BoolProc(False))
#         assert await st.update() is TaskStatus.FAILURE

#     async def test_invalid_return_type_bubbles(self):
#         class _Bad(Process):
#             def delta(self): return TaskStatus.SUCCESS          # misuse
#         with pytest.raises(Exception):
#             await S.BranchState(f=_Bad()).update()



# @pytest.mark.asyncio
# class TestTaskStateLifecycle:
#     """Delegation and caching semantics around the wrapped task."""

#     async def test_running_then_success(self):
#         inner = _ToggleTask()
#         ts = S.TaskState(task=inner)
#         assert await ts.update() is TaskStatus.RUNNING
#         assert await ts.update() is TaskStatus.SUCCESS
#         assert inner.calls == 2                                   # both ticks forwarded

#     async def test_done_task_not_reinvoked(self):
#         class _Done(Action):
#             async def act(self): return TaskStatus.SUCCESS
#         done = _Done(); await done.tick()                         # mark as done
#         ts = S.TaskState(task=done)
#         res = await ts.update();                                       # first call returns SUCCESS
#         print(res)
#         res2 = await ts.update();                                       # second call should not re-tick
#         print(res2)
#         assert ts.task.status is TaskStatus.SUCCESS

