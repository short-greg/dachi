"""Unit tests for CompositeState.

Tests cover composite state lifecycle, child region management, context/post
propagation, and completion logic following the framework testing conventions.
"""

import asyncio
import pytest

from dachi.act._chart._composite import CompositeState
from dachi.act._chart._region import Region
from dachi.act._chart._state import State, FinalState
from dachi.act._chart._base import ChartStatus, InvalidTransition
from dachi.act._chart._event import EventQueue, EventPost
from dachi.core import Scope, ModuleList


@pytest.fixture(autouse=True)
async def cleanup_tasks():
    """Ensure all pending tasks are cancelled after each test."""
    yield
    # Give tasks a moment to complete naturally
    await asyncio.sleep(0.001)
    # Cancel any remaining tasks (except current task)
    tasks = [t for t in asyncio.all_tasks() if not t.done() and t != asyncio.current_task()]
    for task in tasks:
        task.cancel()
    # Wait for cancellation to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


class SimpleState(State):
    async def execute(self, post, **inputs):
        pass


class SimpleFinal(FinalState):
    pass


class SlowState(State):
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.1)


class TestCompositeStateInit:
    """Test __post_init__ method"""

    def test_post_init_calls_parent_init(self):
        composite = CompositeState(regions=ModuleList())
        assert composite._status.get() == ChartStatus.WAITING

    def test_post_init_initializes_tasks_to_empty_list(self):
        composite = CompositeState(regions=ModuleList())
        assert composite._tasks == []

    def test_post_init_initializes_finished_regions_to_empty_set(self):
        composite = CompositeState(regions=ModuleList())
        assert composite._finished_regions == set()

    def test_post_init_sets_status_to_waiting(self):
        composite = CompositeState(regions=ModuleList())
        assert composite.get_status() == ChartStatus.WAITING

    def test_post_init_with_no_regions(self):
        composite = CompositeState(regions=ModuleList())
        assert len(composite.regions) == 0

    def test_post_init_with_single_region(self):
        region = Region(name="child", initial="idle", rules=[])
        composite = CompositeState(regions=ModuleList(items=[region]))
        assert len(composite.regions) == 1

    def test_post_init_with_multiple_regions(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        assert len(composite.regions) == 2

    def test_post_init_inherits_base_state_flags(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        assert composite._termination_requested.get() is False
        assert composite._run_completed.get() is False
        assert composite._executing.get() is False
        assert composite._entered.get() is False
        assert composite._exiting.get() is False


class TestCompositeStateCanRun:
    """Test can_run method"""

    def test_can_run_returns_true_when_entered_and_not_running(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        composite.enter(post, ctx)
        assert composite.can_run() is True

    def test_can_run_returns_false_when_waiting(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        assert composite.can_run() is False

    def test_can_run_returns_false_when_running(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()
        composite.enter(post, ctx)
        composite._executing.set(True)
        assert composite.can_run() is False

    def test_can_run_returns_false_when_completed(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.SUCCESS)
        assert composite.can_run() is False

    def test_can_run_returns_false_when_not_entered(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._entered.set(False)
        assert composite.can_run() is False

    def test_can_run_returns_false_when_failed(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.FAILURE)
        assert composite.can_run() is False


class TestCompositeStateExecute:
    """Test execute method"""

    @pytest.mark.asyncio
    async def test_execute_clears_previous_tasks(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._tasks = [asyncio.create_task(asyncio.sleep(0.01))]
        await composite.execute(post, ctx)
        assert len(composite._tasks) == 1  # New task, old cleared

    @pytest.mark.asyncio
    async def test_execute_creates_task_for_each_region(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        assert len(composite._tasks) == 2

    @pytest.mark.asyncio
    async def test_execute_passes_child_post_to_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue, source=[("main", "composite")])
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        # Verify region started (task created)
        assert len(composite._tasks) == 1

    @pytest.mark.asyncio
    async def test_execute_passes_child_ctx_to_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        # Verify execution started with context
        assert len(composite._tasks) == 1

    @pytest.mark.asyncio
    async def test_execute_registers_finish_callback_for_each_region(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        # Verify callback registered
        assert composite.finish_region in region._finish_callbacks

    @pytest.mark.asyncio
    async def test_execute_returns_none(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        result = await composite.execute(post, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_with_single_region(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        assert len(composite._tasks) == 1

    @pytest.mark.asyncio
    async def test_execute_with_no_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        assert len(composite._tasks) == 0

    @pytest.mark.asyncio
    async def test_execute_multiple_calls_clears_tasks(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        first_task_count = len(composite._tasks)
        await composite.execute(post, ctx)
        assert len(composite._tasks) == first_task_count

    @pytest.mark.asyncio
    async def test_execute_task_creation_order_deterministic(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        await composite.execute(post, ctx)
        # Tasks created in order (2 regions)
        assert len(composite._tasks) == 2

    @pytest.mark.asyncio
    async def test_execute_uses_enumeration_for_child_contexts(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        # Execute should use enumerate to create ctx.child(0), ctx.child(1), etc
        await composite.execute(post, ctx)

        # Verify execution succeeded with 2 tasks (one per region)
        assert len(composite._tasks) == 2


class TestCompositeStateRun:
    """Test run method"""

    @pytest.mark.asyncio
    async def test_run_raises_error_when_cannot_run(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        # Not entered, so cannot run
        with pytest.raises(RuntimeError):
            await composite.run(post, ctx)

    @pytest.mark.asyncio
    async def test_run_completes_immediately_when_no_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await composite.run(post, ctx)
        # Should complete immediately
        assert composite.get_status() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_run_sets_status_to_success_when_no_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await composite.run(post, ctx)
        assert composite._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_run_calls_finish_when_no_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        called = False
        def callback():
            nonlocal called
            called = True

        composite.register_finish_callback(callback)
        composite.enter(post, ctx)
        await composite.run(post, ctx)
        # Finish only called after exit() when exiting=True
        composite.exit(post, ctx)
        await asyncio.sleep(0.01)  # Wait for finish callback
        assert called is True

    @pytest.mark.asyncio
    async def test_run_calls_execute_when_has_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await composite.run(post, ctx)
        # Verify execute was called (tasks created)
        assert len(composite._tasks) == 1

    @pytest.mark.asyncio
    async def test_run_sets_status_to_running(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await composite.run(post, ctx)
        # Status set to RUNNING by run()
        assert composite._status.get() == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_sets_run_completed_to_false_after_execute(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await composite.run(post, ctx)
        assert composite._run_completed.get() is False

    @pytest.mark.asyncio
    async def test_run_does_not_block_waiting_for_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SlowState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        # run() should return immediately (callback-driven)
        await asyncio.wait_for(composite.run(post, ctx), timeout=0.1)

    @pytest.mark.asyncio
    async def test_run_raises_error_when_not_entered(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        with pytest.raises(RuntimeError):
            await composite.run(post, ctx)

    @pytest.mark.asyncio
    async def test_run_raises_error_when_already_completed(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._status.set(ChartStatus.SUCCESS)
        with pytest.raises(RuntimeError):
            await composite.run(post, ctx)


class TestCompositeStateFinishRegion:
    """Test finish_region callback method"""

    @pytest.mark.asyncio
    async def test_finish_region_adds_region_to_finished_set(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)
        await composite.finish_region("child", post, ctx)
        assert "child" in composite._finished_regions

    @pytest.mark.asyncio
    async def test_finish_region_unregisters_callback(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)
        await composite.finish_region("child", post, ctx)
        assert composite.finish_region not in region._finish_callbacks

    @pytest.mark.asyncio
    async def test_finish_region_does_not_finish_when_not_all_complete(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region1.register_finish_callback(composite.finish_region, region1.name, post, ctx)
        region2.register_finish_callback(composite.finish_region, region2.name, post, ctx)

        await composite.finish_region("child1", post, ctx)
        # Should NOT be finished (need child2)
        assert composite.get_status() != ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_finish_region_clears_tasks_when_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        composite._tasks = [asyncio.create_task(asyncio.sleep(0.01))]
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        assert composite._tasks == []

    @pytest.mark.asyncio
    async def test_finish_region_sets_status_to_success_when_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        assert composite._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_finish_region_sets_run_completed_when_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        assert composite._run_completed.get() is True

    @pytest.mark.asyncio
    async def test_finish_region_calls_finish_when_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        called = False
        def callback():
            nonlocal called
            called = True

        composite._exiting.set(True)
        composite.register_finish_callback(callback)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        assert called is True

    @pytest.mark.asyncio
    async def test_finish_region_with_single_region(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        assert composite._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_finish_region_with_multiple_regions_sequential(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region1.register_finish_callback(composite.finish_region, region1.name, post, ctx)
        region2.register_finish_callback(composite.finish_region, region2.name, post, ctx)

        await composite.finish_region("child1", post, ctx)
        assert composite.get_status() != ChartStatus.SUCCESS

        await composite.finish_region("child2", post, ctx)
        assert composite.get_status() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_finish_region_completion_count_correct(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region1.register_finish_callback(composite.finish_region, region1.name, post, ctx)
        region2.register_finish_callback(composite.finish_region, region2.name, post, ctx)

        await composite.finish_region("child1", post, ctx)
        assert len(composite._finished_regions) == 1

        await composite.finish_region("child2", post, ctx)
        assert len(composite._finished_regions) == 2

    @pytest.mark.asyncio
    async def test_finish_region_called_twice_same_region_idempotent(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        # Call again - should be safe
        await composite.finish_region("child", post, ctx)
        assert len(composite._finished_regions) == 1

    @pytest.mark.asyncio
    async def test_finish_region_order_independence(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))
        region2.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite._exiting.set(True)
        region1.register_finish_callback(composite.finish_region, region1.name, post, ctx)
        region2.register_finish_callback(composite.finish_region, region2.name, post, ctx)

        # Finish in reverse order
        await composite.finish_region("child2", post, ctx)
        await composite.finish_region("child1", post, ctx)
        assert composite.get_status() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_finish_region_does_not_finish_when_not_exiting(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        # NOT exiting
        composite._exiting.set(False)
        region.register_finish_callback(composite.finish_region, region.name, post, ctx)

        await composite.finish_region("child", post, ctx)
        # Should NOT finish because _exiting is False
        assert composite.get_status() != ChartStatus.SUCCESS


class TestCompositeStateReset:
    """Test reset method"""

    def test_reset_calls_parent_reset(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.SUCCESS)
        composite.reset()
        assert composite._status.get() == ChartStatus.WAITING

    @pytest.mark.asyncio
    async def test_reset_clears_tasks(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.SUCCESS)
        # Create a real task in async context
        composite._tasks = [asyncio.create_task(asyncio.sleep(0.01))]
        composite.reset()
        assert composite._tasks == []

    def test_reset_clears_finished_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.SUCCESS)
        composite._finished_regions = {"child1", "child2"}
        composite.reset()
        assert composite._finished_regions == set()

    def test_reset_raises_error_when_running(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.RUNNING)
        with pytest.raises(InvalidTransition):
            composite.reset()

    def test_reset_works_after_success(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        composite._status.set(ChartStatus.SUCCESS)
        composite.reset()
        assert composite._status.get() == ChartStatus.WAITING


class TestCompositeStateExit:
    """Test exit method"""

    @pytest.mark.asyncio
    async def test_exit_raises_error_when_cannot_exit(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        # Not entered, cannot exit
        with pytest.raises(InvalidTransition):
            composite.exit(post, ctx)

    @pytest.mark.asyncio
    async def test_exit_sets_exiting_flag(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        composite.exit(post, ctx)
        assert composite._exiting.get() is True

    @pytest.mark.asyncio
    async def test_exit_sets_termination_requested(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        composite.exit(post, ctx)
        assert composite._termination_requested.get() is True

    @pytest.mark.asyncio
    async def test_exit_stops_running_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        # Start region
        await region.start(post.child("child"), ctx.child(0))

        composite.exit(post, ctx)
        # Wait for async stop task to execute
        await asyncio.sleep(0.01)
        # Region should be stopped (or stopping)
        assert region.status.is_preempting() or region.status.is_completed()

    @pytest.mark.asyncio
    async def test_exit_with_empty_composite(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)

        composite.exit(post, ctx)
        # Empty composite should succeed
        assert composite._status.get() == ChartStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_exit_sets_status_to_preempting_when_not_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SlowState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        # Start region (will be running)
        await region.start(post.child("child"), ctx.child(0))

        composite.exit(post, ctx)
        assert composite._status.get() == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_exit_calls_finish_for_empty_composite(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        called = False
        def callback():
            nonlocal called
            called = True

        composite.register_finish_callback(callback)
        composite.enter(post, ctx)

        composite.exit(post, ctx)
        # Wait for finish callback to be called (scheduled as task)
        await asyncio.sleep(0.01)
        assert called is True

    @pytest.mark.asyncio
    async def test_exit_does_not_call_finish_when_not_all_complete(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SlowState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        called = False
        def callback():
            nonlocal called
            called = True

        composite.register_finish_callback(callback)
        composite.enter(post, ctx)
        await region.start(post.child("child"), ctx.child(0))

        composite.exit(post, ctx)
        assert called is False

    @pytest.mark.asyncio
    async def test_exit_unregisters_callbacks_for_running_regions(self):
        region = Region(name="child", initial="idle", rules=[])
        region.add(SlowState(name="idle"))
        composite = CompositeState(regions=ModuleList(items=[region]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await region.start(post.child("child"), ctx.child(0))

        # Register callback
        region.register_finish_callback(composite.finish_region, "child", post, ctx)

        composite.exit(post, ctx)
        # Callback should be unregistered
        assert composite.finish_region not in region._finish_callbacks

    @pytest.mark.asyncio
    async def test_exit_with_multiple_regions_some_complete(self):
        # Use SimpleState for region1 instead of FinalState
        # Region1 will complete quickly, region2 will still be running
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])
        region1.add(SimpleState(name="idle"))  # Quick to complete
        region2.add(SlowState(name="idle"))     # Still running
        composite = CompositeState(regions=ModuleList(items=[region1, region2]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)
        await region1.start(post.child("child1"), ctx.child(0))
        await region2.start(post.child("child2"), ctx.child(1))

        # Wait for region1 to complete naturally
        await asyncio.sleep(0.01)

        # Now region1 should be completed, region2 still running
        composite.exit(post, ctx)
        # One complete, one not - should be preempting
        assert composite._status.get() == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_exit_with_no_regions(self):
        composite = CompositeState(regions=ModuleList(items=[]))
        queue = EventQueue()
        post = EventPost(queue=queue)
        scope = Scope()
        ctx = scope.ctx()

        composite.enter(post, ctx)

        composite.exit(post, ctx)
        # Empty composite should succeed on exit
        assert composite._status.get() == ChartStatus.SUCCESS


class TestCompositeStateRestrictedSchema:
    """Test CompositeState.restricted_schema() - Pattern A (Pass-Through)"""

    def test_restricted_schema_returns_unrestricted_when_states_none(self):
        """Test that states=None returns unrestricted schema"""
        composite = CompositeState(regions=ModuleList(items=[]))
        restricted = composite.restricted_schema(states=None)
        unrestricted = composite.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_passes_states_to_region(self):
        """Test that states are passed through to Region schema"""
        composite = CompositeState(regions=ModuleList(items=[]))

        # Restrict to only SimpleState and SimpleFinal
        restricted = composite.restricted_schema(
            states=[SimpleState, SimpleFinal]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        # Region schema should have Allowed_states
        assert "Allowed_states" in restricted["$defs"]

        # Check that Region schema is in defs
        region_spec_keys = [k for k in restricted["$defs"].keys() if "RegionSpec" in k]
        assert len(region_spec_keys) >= 1

    def test_restricted_schema_updates_regions_field(self):
        """Test that regions field is updated with restricted Region schema"""
        composite = CompositeState(regions=ModuleList(items=[]))

        restricted = composite.restricted_schema(
            states=[SimpleState, SimpleFinal]
        )

        # regions field should have items pointing to a Region schema
        regions_schema = restricted["properties"]["regions"]
        assert "items" in regions_schema

        # The items should be a $ref to a RegionSpec
        items = regions_schema["items"]
        assert "$ref" in items
        assert "RegionSpec" in items["$ref"]

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared'"""
        composite = CompositeState(regions=ModuleList(items=[]))
        restricted = composite.restricted_schema(states=[SimpleState])

        # Should use shared union in $defs
        assert "Allowed_states" in restricted["$defs"]

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        composite = CompositeState(regions=ModuleList(items=[]))
        restricted = composite.restricted_schema(
            states=[SimpleState, SimpleFinal],
            _profile="inline"
        )

        # Should still have defs for the individual states
        defs_keys = restricted["$defs"].keys()
        assert any("SimpleStateSpec" in key for key in defs_keys)
        assert any("SimpleFinalSpec" in key for key in defs_keys)
