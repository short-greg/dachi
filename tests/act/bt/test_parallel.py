"""Updated behaviour‑tree unit tests.

These tests have been adapted to the new asynchronous task execution model and
`BaseModule` keyword‑initialisation.  **No tests were added or removed – every
original case remains, simply modernised.**

Google‑style docstrings and minimal comments are retained per project
conventions.  All async tests use `pytest.mark.asyncio`.
"""

import pytest
from dachi.core import ModuleList
from dachi.act._bt._core import TaskStatus

from dachi.act._bt._parallel import MultiTask
from ..utils import ImmediateAction, SetStorageAction, create_test_ctx


class TestParallelValidate:
    """`Parallel.validate` should enforce quorum invariants and raise early."""

    def _parallel(self, fails: int, succ: int, n: int = 3):
        tasks = ModuleList(vals=[ImmediateAction(status_val=TaskStatus.RUNNING) for _ in range(n)])
        return MultiTask(tasks=tasks, fails_on=fails, succeeds_on=succ)

    def test_ok_when_thresholds_within_bounds(self):
        # Should not raise during construction (validation happens automatically)
        par = self._parallel(fails=1, succ=-1)  # always succeed with *all* successes
        assert par is not None

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
        par = MultiTask(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS) for _ in range(3)], succeeds_on=-1, fails_on=1)
        assert await par.tick(create_test_ctx()) is TaskStatus.SUCCESS

    async def test_failure_threshold(self):
        par = MultiTask(tasks=[ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)], fails_on=1, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.RUNNING

    async def test_running_until_quorum(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)]
        par = MultiTask(tasks=tasks, fails_on=2, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.RUNNING

    async def test_fails_on_1_failure(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE)]
        par = MultiTask(tasks=tasks, fails_on=1, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.FAILURE


class TestMultiTaskSerialization:
    """MultiTask serialization and generic type parameter tests."""

    def test_to_spec_preserves_multitask_structure(self):
        """to_spec() preserves MultiTask structure."""
        task1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        task2 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        original = MultiTask[ImmediateAction](tasks=[task1, task2], fails_on=1, succeeds_on=2)

        spec = original.to_spec()

        assert "MultiTask" in spec["KIND"]
        assert "tasks" in spec
        assert len(spec["tasks"]) == 2
        assert spec["fails_on"] == 1
        assert spec["succeeds_on"] == 2

    def test_to_spec_preserves_generic_type_parameter(self):
        """to_spec() preserves MultiTask generic type parameter."""
        task1 = SetStorageAction(value=10)
        task2 = SetStorageAction(value=20)
        original = MultiTask[SetStorageAction](tasks=[task1, task2])

        spec = original.to_spec()

        assert "MultiTask" in spec["KIND"]
        assert spec["tasks"]["vals"][0]["value"] == 10
        assert spec["tasks"]["vals"][1]["value"] == 20

    def test_spec_roundtrip_with_union_type(self):
        """Spec round-trip with MultiTask[Task1 | Task2] works."""
        task1 = ImmediateAction(status_val=TaskStatus.SUCCESS)
        task2 = SetStorageAction(value=42)
        original = MultiTask[ImmediateAction | SetStorageAction](
            tasks=[task1, task2],
            fails_on=1,
            succeeds_on=2
        )

        spec = original.to_spec()
        restored = MultiTask[ImmediateAction | SetStorageAction].from_spec(spec)

        assert len(restored.tasks) == 2
        assert isinstance(restored.tasks[0], ImmediateAction)
        assert isinstance(restored.tasks[1], SetStorageAction)
        assert restored.tasks[0].status_val == TaskStatus.SUCCESS
        assert restored.tasks[1].value == 42
        assert restored.fails_on == 1
        assert restored.succeeds_on == 2
