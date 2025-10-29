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

from dachi.act._bt._parallel import Multi
from .utils import ImmediateAction, SetStorageAction, create_test_ctx


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
        assert await par.tick(create_test_ctx()) is TaskStatus.SUCCESS

    async def test_failure_threshold(self):
        par = Multi(tasks=[ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)], fails_on=1, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.RUNNING

    async def test_running_until_quorum(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE), ImmediateAction(status_val=TaskStatus.RUNNING)]
        par = Multi(tasks=tasks, fails_on=2, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.RUNNING

    async def test_fails_on_1_failure(self):
        tasks = [ImmediateAction(status_val=TaskStatus.SUCCESS), ImmediateAction(status_val=TaskStatus.FAILURE)]
        par = Multi(tasks=tasks, fails_on=1, succeeds_on=2)
        assert await par.tick(create_test_ctx()) is TaskStatus.FAILURE


class TestMultiRestrictedSchema:
    """Test Multi.restricted_schema() - Pattern B (Direct Variants)"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)
        restricted = multi.restricted_schema(tasks=None)
        unrestricted = multi.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_tasks_field_with_variants(self):
        """Test that tasks field is restricted to specified variants"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)

        # Restrict to only ImmediateAction and SetStorageAction
        restricted = multi.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        assert "Allowed_TaskSpec" in restricted["$defs"]

        # Check that Allowed_TaskSpec contains our variants
        allowed_union = restricted["$defs"]["Allowed_TaskSpec"]
        assert "oneOf" in allowed_union
        refs = allowed_union["oneOf"]
        assert len(refs) == 2

        # Extract spec names from refs
        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        # Spec names include module path, so check if they contain the expected name
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("SetStorageActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)
        restricted = multi.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_TaskSpec" in restricted["$defs"]

        # tasks field should reference the shared union
        # Handle nullable field (anyOf with null)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert option["items"] == {"$ref": "#/$defs/Allowed_TaskSpec"}
                    break
        else:
            # Non-nullable
            assert tasks_schema["items"] == {"$ref": "#/$defs/Allowed_TaskSpec"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)
        restricted = multi.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Should still have defs for the individual tasks (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)

        # But tasks field should have inline oneOf (no Allowed_TaskSpec)
        tasks_schema = restricted["properties"]["tasks"]
        if "anyOf" in tasks_schema:
            # Nullable: find the array option
            for option in tasks_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    assert "oneOf" in option["items"]
                    break
        else:
            # Non-nullable
            assert "oneOf" in tasks_schema["items"]

    def test_restricted_schema_with_task_spec_class(self):
        """Test that TaskSpec classes work as variants"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)

        # Get the spec class
        spec_class = ImmediateAction.schema_model()

        restricted = multi.restricted_schema(tasks=[spec_class])

        # Should work and include the task (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)

    def test_restricted_schema_with_mixed_formats(self):
        """Test that mixed variant formats work together"""
        multi = Multi(tasks=[ImmediateAction(status_val=TaskStatus.SUCCESS)], fails_on=1, succeeds_on=1)

        # Mix: Task class, TaskSpec class, and schema dict
        action_spec = SetStorageAction.schema_model()
        immediate_schema = ImmediateAction.schema()

        restricted = multi.restricted_schema(
            tasks=[
                ImmediateAction,  # Task class
                action_spec,       # Spec class
                immediate_schema   # Schema dict (will be duplicate)
            ]
        )

        # Should deduplicate and work correctly (with full module path)
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)
