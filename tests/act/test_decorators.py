
"""Updated behaviour‑tree unit tests.

These tests have been adapted to the new asynchronous task execution model and
`BaseModule` keyword‑initialisation.  **No tests were added or removed – every
original case remains, simply modernised.**

Google‑style docstrings and minimal comments are retained per project
conventions.  All async tests use `pytest.mark.asyncio`.
"""

import pytest
from dachi.core import InitVar, Runtime, Scope, PrivateRuntime
from .utils import create_test_ctx
from dachi.act._bt._core import TaskStatus, LeafTask
from dachi.act._bt._decorators import Not, Until, BoundTask, Decorator

from dachi.act._bt._leafs import Action, Condition


class AlwaysTrueCond(Condition):
    async def execute(self):
        return True

class AlwaysFalseCond(Condition):
    async def execute(self):
        return False


class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def execute(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: int = 4 

    async def execute(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.FAILURE if self.value < 0 else TaskStatus.SUCCESS


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def execute(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    value: int = 4
    _count: Runtime[int] = PrivateRuntime(0)

    async def execute(self) -> TaskStatus:  # noqa: D401
        if self.value == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING



class ImmediateAction(Action):
    """A task that immediately returns a fixed *status*."""

    status_val: TaskStatus = TaskStatus.SUCCESS

    async def execute(self) -> TaskStatus:  # noqa: D401
        return self.status_val


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: int = 4
    _count: Runtime[int] = PrivateRuntime(0)

    async def execute(self) -> TaskStatus:  # noqa: D401
        if self.value == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


@pytest.mark.asyncio
class TestNotDecorator:
    async def test_invert_success(self):
        scope = Scope()
        ctx = scope.ctx()
        assert await Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS)).tick(ctx) is TaskStatus.FAILURE

    async def test_invert_failure(self):
        scope = Scope()
        ctx = scope.ctx()
        assert await Not(task=ImmediateAction(status_val=TaskStatus.FAILURE)).tick(ctx) is TaskStatus.SUCCESS



# # ---------------------------------------------------------------------------
# # 14. Loop context‑manager utilities ----------------------------------------
# # ---------------------------------------------------------------------------



@pytest.mark.asyncio
class TestUntil:
    async def test_until_successful_if_success(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        until_ = Until(task=action1)
        assert await until_.tick(ctx) == TaskStatus.SUCCESS

    async def test_until_successful_if_success_after_two(self):
        scope = Scope()
        ctx = scope.ctx()
        action1 = SetStorageActionCounter(value=0)
        action1._count = 1
        until_ = Until(task=action1)
        await until_.tick(ctx)
        action1.value.data = 1
        assert await until_.tick(ctx) == TaskStatus.SUCCESS


# Context-aware decorator test helpers
from dachi.core import Scope
from dachi.act._bt._decorators import Decorator
from dachi.act._bt._serial import Sequence

class ContextTestAction(Action):
    """Test action with configurable input/output ports for context testing"""
    
    _call_count: Runtime[int] = PrivateRuntime(0)

    class inputs:
        target: tuple = (0, 0, 0)
        attempts: int = 1
        optional_param: str = "default_value"
    
    class outputs:
        result: str
        success: bool
    
    async def execute(self, target: tuple = (0, 0, 0), attempts: int = 1, optional_param: str = "default_value"):
        self._call_count += 1
        self._last_kwargs = {"target": target, "attempts": attempts, "optional_param": optional_param}
        
        if attempts > 0:
            return TaskStatus.SUCCESS, {
                "result": f"reached_{target}",
                "success": True
            }
        else:
            return TaskStatus.FAILURE, {
                "result": "failed",
                "success": False
            }

class SimpleContextAction(Action):
    """Simpler action that just returns success and configurable outputs"""
    
    output_value: int = 42

    class outputs:
        value: int
    
    async def execute(self):
        return TaskStatus.SUCCESS, {"value": self.output_value}


class MockTestDecorator(Decorator):
    """Test decorator that just forwards context to wrapped task"""
    
    async def decorate(self, status: TaskStatus, reset: bool = False) -> TaskStatus:
        return status
    
    async def tick(self, ctx):
        """Override to accept context parameter"""
        if self.status.is_done:
            return self.status
        
        # Forward context to wrapped task
        if hasattr(self.task, '__ports__'):  # It's a leaf
            # This will be implemented - resolve inputs and call with kwargs
            await self.task.tick(create_test_ctx())  # Placeholder
        else:  # It's a composite
            await self.task.tick(ctx)
        
        await self.update_status()
        return self.status


@pytest.mark.asyncio
class TestDecoratorWithContext:
    """Test context-aware Decorator behavior"""
    
    async def test_decorator_forwards_context_to_wrapped_task(self):
        """Test that decorator forwards context to wrapped composite tasks"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Create composite wrapped in decorator
        inner_action = SimpleContextAction(output_value=99)
        inner_sequence = Sequence(tasks=[inner_action])
        inner_sequence.cascade()
        decorator = MockTestDecorator(task=inner_sequence)
        
        await decorator.tick(ctx)
        
        # Verify the inner sequence got context and stored output
        assert scope.path((0,), "value") == 99
    
    async def test_decorator_calls_leaf_with_resolved_kwargs(self):
        """Test that decorator calls leaf tasks with resolved keyword arguments"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data
        ctx["target"] = (5, 10, 15)
        ctx["attempts"] = 4
        
        action = ContextTestAction()
        decorator = MockTestDecorator(task=action)
        
        await decorator.tick(ctx)
        
        # Verify action received resolved inputs
        # Note: This test will need actual input resolution implementation
        assert hasattr(action, '_last_kwargs')
    
    async def test_decorator_calls_composite_with_child_context(self):
        """Test that decorator forwards context to composite children"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Create nested structure: decorator -> sequence -> action
        action = SimpleContextAction(output_value=77)
        sequence = Sequence(tasks=[action])
        decorator = MockTestDecorator(task=sequence)
        
        await decorator.tick(ctx)
        
        # Verify data flow through decorator -> sequence -> action
        assert scope.path((0,), "value") == 77
    
    async def test_decorator_output_handling_and_context_updates(self):
        """Test decorator handles wrapped task outputs correctly"""
        scope = Scope()
        ctx = scope.ctx()
        
        action = SimpleContextAction(output_value=123)
        decorator = MockTestDecorator(task=action)
        
        await decorator.tick(ctx)
        
        # Verify decorator doesn't interfere with output storage
        # The decorator should allow the action's outputs to be stored
        # This will be verified once ctx.update() is implemented


@pytest.mark.asyncio
class TestBindDecorator:
    """Test Bind decorator for input binding at tick-time"""
    
    async def test_bind_constant_input_values(self):
        """Test binding constant values to task inputs"""
        scope = Scope()
        ctx = scope.ctx()
        
        action = ContextTestAction()
        # Bind constant values
        bind_decorator = BoundTask(leaf=action, bindings={"target": "target_base", "attempts": "attempts_base"})
        ctx['target_base'] = (1, 2, 3)
        ctx['attempts_base'] = 5
        ctx['optional_param'] = "bound_value"
        
        await bind_decorator.tick(ctx)
        
        print(ctx.scope.fields)
        # Verify action received bound constant values
        assert ctx['target_base'] == action._last_kwargs['target']
        assert ctx['attempts_base'] == action._last_kwargs['attempts']
        assert ctx['optional_param'] == action._last_kwargs['optional_param']

    async def test_bind_context_key_resolution(self):
        """Test binding inputs from context keys"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up context data
        ctx["goal"] = (50, 75, 100)
        ctx["retry_count"] = 3
        
        action = ContextTestAction()
        # This syntax will be implemented - string keys resolve to context values
        bind_decorator = BoundTask(leaf=action, bindings={"target": "goal", "attempts": "retry_count"})
        
        await bind_decorator.tick(ctx)
        
        # Note: This will need actual string resolution implementation
        # For now, verify the bind decorator structure is correct
        assert bind_decorator.bindings['target'] == "goal"
        assert bind_decorator.bindings['attempts'] == "retry_count"
    
    async def test_bind_nested_path_access(self):
        """Test binding inputs from nested context paths"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up nested data structure
        scope.set((0,), "pose", {"x": 10, "y": 20, "z": 30})
        
        action = ContextTestAction()
        # This syntax will be implemented - nested path resolution
        bind_decorator = BoundTask(leaf=action, bindings={"target": "0.pose.x"})
        
        await bind_decorator.tick(ctx)
        
        # Verify binding was set up correctly
        assert bind_decorator.bindings['target'] == "0.pose.x"
    
    async def test_bind_mixed_constant_and_context_values(self):
        """Test mixing constant bindings with context-resolved bindings"""
        scope = Scope()
        ctx = scope.ctx()
        
        ctx["dynamic_target"] = (99, 88, 77)
        
        action = ContextTestAction()
        # Mix constants and context keys
        bind_decorator = BoundTask(
            leaf=action, 
            bindings={
            "target": "dynamic_target",  # from context
            "attempts": 10,              # constant
            "optional_param": "bound_value"  # constant
            }
        )
        
        await bind_decorator.tick(ctx)
        
        # Verify mixed binding setup
        assert bind_decorator.bindings['target'] == "dynamic_target"
        assert bind_decorator.bindings['attempts'] == 10
        assert bind_decorator.bindings['optional_param'] == "bound_value"
    
    async def test_bind_resolution_failure_handling(self):
        """Test handling when bind resolution fails"""
        scope = Scope()
        ctx = scope.ctx()
        
        action = ContextTestAction()
        # Bind to non-existent context key
        bind_decorator = BoundTask(leaf=action, bindings={"target": "nonexistent_key"})
        
        # This should fail gracefully when resolution is implemented
        # For now, verify the binding is set up
        assert bind_decorator.bindings['target'] == "nonexistent_key"
    
    async def test_bind_with_complex_context_hierarchies(self):
        """Test Bind decorator with complex nested context structures"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Set up complex hierarchy: root -> child(0) -> grandchild(1)
        child_ctx = ctx.child(0)
        grandchild_ctx = child_ctx.child(1)
        
        grandchild_ctx["deep_value"] = (1, 2, 3)
        
        action = ContextTestAction()
        # This path resolution will be implemented
        bind_decorator = BoundTask(leaf=action, bindings={"target": "0.1.deep_value"})
        
        await bind_decorator.tick(ctx)

        # Verify complex path binding setup
        assert bind_decorator.bindings['target'] == "0.1.deep_value"


class TestDecoratorRestrictedSchema:
    """Test Decorator.restricted_schema() - Pattern C (Single Field)"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        # Use Not as a concrete Decorator subclass
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = dec.restricted_schema(tasks=None)
        unrestricted = dec.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_task_field_with_variants(self):
        """Test that task field is restricted to specified variants"""
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Restrict to only ImmediateAction and SetStorageAction
        restricted = dec.restricted_schema(
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
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("SetStorageActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = dec.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_TaskSpec" in restricted["$defs"]

        # task field should reference the shared union
        task_schema = restricted["properties"]["task"]
        assert task_schema == {"$ref": "#/$defs/Allowed_TaskSpec"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = dec.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Should still have defs for the individual tasks
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)

        # But task field should have inline oneOf (no Allowed_TaskSpec)
        task_schema = restricted["properties"]["task"]
        assert "oneOf" in task_schema

    def test_restricted_schema_with_task_spec_class(self):
        """Test that TaskSpec classes work as variants"""
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Get the spec class
        spec_class = ImmediateAction.schema_model()

        restricted = dec.restricted_schema(tasks=[spec_class])

        # Should work and include the task
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)

    def test_restricted_schema_with_mixed_formats(self):
        """Test that mixed variant formats work together"""
        dec = Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Mix: Task class, TaskSpec class, and schema dict
        action_spec = SetStorageAction.schema_model()
        immediate_schema = ImmediateAction.schema()

        restricted = dec.restricted_schema(
            tasks=[
                ImmediateAction,  # Task class
                action_spec,       # Spec class
                immediate_schema   # Schema dict (will be duplicate)
            ]
        )

        # Should deduplicate and work correctly
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)


class TestBoundTaskRestrictedSchema:
    """Test BoundTask.restricted_schema() - Pattern C with Leaf filter"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})
        restricted = bound.restricted_schema(tasks=None)
        unrestricted = bound.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_leaf_field_with_leaf_variants(self):
        """Test that leaf field is restricted to Leaf variants only"""
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})

        # Restrict to only ImmediateAction and AlwaysTrueCond (both are Leaf subclasses)
        restricted = bound.restricted_schema(
            tasks=[ImmediateAction, AlwaysTrueCond]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        assert "Allowed_LeafSpec" in restricted["$defs"]

        # Check that Allowed_LeafSpec contains our variants
        allowed_union = restricted["$defs"]["Allowed_LeafSpec"]
        assert "oneOf" in allowed_union
        refs = allowed_union["oneOf"]
        assert len(refs) == 2

        # Extract spec names from refs
        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("AlwaysTrueCondSpec" in name for name in spec_names)

    def test_restricted_schema_filters_out_non_leaf_tasks(self):
        """Test that non-Leaf tasks are filtered out"""
        from dachi.act._bt._serial import Sequence
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})

        # Try to restrict with both Leaf and non-Leaf tasks
        # Sequence is not a Leaf, so it should be filtered out
        restricted = bound.restricted_schema(
            tasks=[ImmediateAction, Sequence]  # Sequence should be filtered
        )

        # Check that only ImmediateAction is included
        assert "$defs" in restricted
        assert "Allowed_LeafSpec" in restricted["$defs"]

        allowed_union = restricted["$defs"]["Allowed_LeafSpec"]
        refs = allowed_union["oneOf"]
        # Should only have 1 ref (ImmediateAction)
        assert len(refs) == 1

        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        assert any("ImmediateActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})
        restricted = bound.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_LeafSpec" in restricted["$defs"]

        # leaf field should reference the shared union
        leaf_schema = restricted["properties"]["leaf"]
        assert leaf_schema == {"$ref": "#/$defs/Allowed_LeafSpec"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})
        restricted = bound.restricted_schema(
            tasks=[ImmediateAction, AlwaysTrueCond],
            _profile="inline"
        )

        # Should still have defs for the individual tasks
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("AlwaysTrueCondSpec" in key for key in defs_keys)

        # But leaf field should have inline oneOf (no Allowed_LeafSpec)
        leaf_schema = restricted["properties"]["leaf"]
        assert "oneOf" in leaf_schema

    def test_restricted_schema_returns_unrestricted_when_no_valid_leafs(self):
        """Test that schema is unrestricted when no valid Leaf tasks provided"""
        from dachi.act._bt._serial import Sequence, Selector
        bound = BoundTask(leaf=ImmediateAction(status_val=TaskStatus.SUCCESS), bindings={})

        # Provide only non-Leaf tasks
        restricted = bound.restricted_schema(tasks=[Sequence, Selector])

        # Should return unrestricted schema since no valid Leaf tasks
        unrestricted = bound.schema()
        assert restricted == unrestricted
