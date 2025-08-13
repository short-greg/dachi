"""Updated behaviour‑tree unit tests.

These tests have been adapted to the new asynchronous task execution model and
`BaseModule` keyword‑initialisation.  **No tests were added or removed – every
original case remains, simply modernised.**

Google‑style docstrings and minimal comments are retained per project
conventions.  All async tests use `pytest.mark.asyncio`.
"""

import pytest
from dachi.core import ModuleList
from dachi.act._core import TaskStatus

from dachi.act._parallel import Multi
from .utils import ImmediateAction


class TestParallelValidate:
    """`Parallel.validate` should enforce quorum invariants and raise early."""

    def _parallel(self, fails: int, succ: int, n: int = 3):
        tasks = ModuleList(items=[ImmediateAction(status_val=TaskStatus.RUNNING) for _ in range(n)])
        return Multi(tasks=tasks, fails_on=fails, succeeds_on=succ)

    def test_ok_when_thresholds_within_bounds(self):
        par = self._parallel(fails=1, succ=-1)  # always succeed with *all* successes
        # Should not raise
        par.validate()

    def test_error_when_threshold_exceeds_task_count(self):
        # 2 + 3 - 1 = 4 > 3 triggers the guard in constructor
        with pytest.raises(ValueError):
            self._parallel(fails=2, succ=3, n=3)

    def test_error_when_zero_threshold(self):
        with pytest.raises(ValueError):
            self._parallel(fails=0, succ=1)



@pytest.mark.asyncio
class TestParallel:
    async def test_all_success(self):
        par = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS) for _ in range(3)], succeeds_on=-1, fails_on=1)
        assert await par.tick() is TaskStatus.SUCCESS

    async def test_failure_threshold(self):
        par = Multi(tasks=[ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)], fails_on=1, succeeds_on=2)
        assert await par.tick() is TaskStatus.RUNNING

    async def test_running_until_quorum(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)]
        par = Multi(tasks=tasks, fails_on=2, succeeds_on=2)
        assert await par.tick() is TaskStatus.RUNNING

    async def test_fails_on_1_failure(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE)]
        par = Multi(tasks=tasks, fails_on=1, succeeds_on=2)
        assert await par.tick() is TaskStatus.FAILURE
