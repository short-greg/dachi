# """Updated behaviour‑tree unit tests.

# These tests have been adapted to the new asynchronous task execution model and
# `BaseModule` keyword‑initialisation.  **No tests were added or removed – every
# original case remains, simply modernised.**

# Google‑style docstrings and minimal comments are retained per project
# conventions.  All async tests use `pytest.mark.asyncio`.
# """

# import asyncio
# import types
# import random
# import pytest
# from dachi.core import InitVar, Attr
# from dachi.act._core import TaskStatus
# from dachi.core import ModuleList, Attr
# from dachi.act._leafs import Action, Condition, WaitCondition


# class ATask(Action):
#     """Always succeeds – used to stub out generic actions."""
#     x: int = 1

#     async def act(self) -> TaskStatus:  # noqa: D401
#         return TaskStatus.SUCCESS


# class SetStorageAction(Action):
#     """Action whose success/failure depends on *value*."""

#     value: InitVar[int] = 4

#     def __post_init__(self, value: int):
#         # TODO: enforce post init is called
#         super().__post_init__()
#         self.value = Attr[int](value)

#     async def act(self) -> TaskStatus:  # noqa: D401
#         print('Acting!')
#         return TaskStatus.FAILURE if self.value.data < 0 else TaskStatus.SUCCESS


# class SampleCondition(Condition):
#     """Condition – true if *x* is non‑negative."""

#     x: int = 1

#     async def condition(self) -> bool:  # noqa: D401
#         return self.x >= 0


# class SetStorageActionCounter(Action):
#     """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

#     # __store__ = ["value"]
#     value: InitVar[int] = 4

#     def __post_init__(self, value: int=4):
#         super().__post_init__()
#         self._count = 0
#         self.value = Attr[int](value)

#     async def act(self) -> TaskStatus:  # noqa: D401
#         if self.value.data == 0:
#             return TaskStatus.FAILURE
#         self._count += 1
#         if self._count == 2:
#             return TaskStatus.SUCCESS
#         if self._count < 0:
#             return TaskStatus.FAILURE
#         return TaskStatus.RUNNING


# class _ImmediateAction(Action):
#     """A task that immediately returns a preset *status_val*."""

#     status_val: InitVar[TaskStatus]

#     def __post_init__(self, status_val: TaskStatus):
#         super().__post_init__()
#         self._ret = status_val

#     async def act(self) -> TaskStatus:
#         return self._ret


# class _FlagWaitCond(WaitCondition):
#     """WaitCondition whose outcome is controlled via the *flag* attribute."""

#     flag: bool = True

#     async def condition(self) -> bool:
#         return self.flag


# ---------------------------------------------------------------------------
# Tests start here – each class targets a single public surface.
# ---------------------------------------------------------------------------



#     # async def test_reset_propagates(self):
#     #     inner = SequenceAction([TaskStatus.RUNNING, TaskStatus.SUCCESS])
#     #     par = behavior.Parallel(tasks=[inner], fails_on=1, succeeds_on=1)
#     #     await par.tick(); par.reset()
#     #     assert inner.status is TaskStatus.READY



# # ---------------------------------------------------------------------------
# # 15. WaitCondition ----------------------------------------------------------
# # ---------------------------------------------------------------------------


# # ---------------------------------------------------------------------------
# # 16. CountLimit -------------------------------------------------------------
# # ---------------------------------------------------------------------------


# # ---------------------------------------------------------------------------
# # 17. PreemptCond ------------------------------------------------------------
# # ---------------------------------------------------------------------------



# @pytest.mark.asyncio
# class TestRunTask:
#     async def test_run_task_yields(self):
#         act = SequenceAction([TaskStatus.RUNNING, TaskStatus.SUCCESS])
#         collected = [s async for s in behavior.run_task(act, interval=None)]
#         assert collected == [TaskStatus.RUNNING, TaskStatus.SUCCESS]
#         assert act.status is TaskStatus.SUCCESS

