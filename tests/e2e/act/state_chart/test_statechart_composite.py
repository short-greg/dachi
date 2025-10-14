"""Composite State E2E tests for StateChart.

These tests validate CompositeState functionality:
- Parallel region execution
- Composite state completion when all regions finish
- Data isolation between parallel regions
- Nested composite states
- Composite states within larger state machines

Each test demonstrates realistic parallel execution patterns.
"""

import asyncio
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from dachi.act._chart._chart import StateChart, ChartStatus
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.act._chart._composite import CompositeState
from dachi.core import Scope, ModuleList


pytestmark = pytest.mark.e2e


# ============================================================================
# Helper Utilities
# ============================================================================

async def wait_for_chart(chart: StateChart, timeout: float = 2.0) -> bool:
    """Wait for chart to complete with timeout."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        status = chart._status.get()
        if status.is_completed():
            return True
        await asyncio.sleep(0.01)
    return False


# ============================================================================
# Test 1: Basic Parallel Regions
# ============================================================================

class TaskAState(State):
    """Task A that runs independently."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.05)
        await post.aforward("task_a_done")
        return {"task_a": "completed"}


class TaskBState(State):
    """Task B that runs independently."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.03)
        await post.aforward("task_b_done")
        return {"task_b": "completed"}


class TaskDoneState(FinalState):
    """Task completion."""
    pass


class TestBasicParallelRegions:
    """Test basic parallel region execution in composite state."""

    @pytest.mark.asyncio
    async def test_two_regions_run_in_parallel(self):
        """Test composite state runs two regions in parallel."""
        # Create two parallel regions
        region_a = Region(name="region_a", initial="task_a", rules=[
            Rule(event_type="task_a_done", target="SUCCESS"),
        ])
        region_a["task_a"] = TaskAState()

        region_b = Region(name="region_b", initial="task_b", rules=[
            Rule(event_type="task_b_done", target="SUCCESS"),
        ])
        region_b["task_b"] = TaskBState()

        # Create composite state with parallel regions
        composite = CompositeState(
            name="parallel_tasks",
            regions=ModuleList(items=[region_a, region_b])
        )

        # Wrapper region to hold composite
        # Add rule to transition composite to SUCCESS when both tasks done
        main_region = Region(name="main", initial="parallel_tasks", rules=[
            Rule(event_type="all_tasks_done", target="SUCCESS"),
        ])
        main_region["parallel_tasks"] = composite

        chart = StateChart(name="parallel_test", regions=[main_region])
        await chart.start()

        # Wait for both regions to complete
        await asyncio.sleep(0.1)

        # Both child regions should have completed
        assert region_a.current_state_name == "SUCCESS"
        assert region_b.current_state_name == "SUCCESS"

        # Composite should be done running (but still waiting for transition event)
        assert composite._run_completed.get() == True

        # Post event to transition composite
        chart.post("all_tasks_done")
        await asyncio.sleep(0.05)

        # Now main region should have transitioned to SUCCESS
        assert main_region.current_state_name == "SUCCESS"


# ============================================================================
# Test 2: Composite with Sequential Child Workflow
# ============================================================================

class Step1State(State):
    """First step in workflow."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("step1_done")
        return {"step": 1}


class Step2State(State):
    """Second step in workflow."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("step2_done")
        return {"step": 2}


class WorkflowComplete(FinalState):
    """Workflow complete."""
    pass


class TestSequentialChildWorkflow:
    """Test composite containing region with sequential workflow."""

    @pytest.mark.asyncio
    async def test_child_region_executes_sequential_workflow(self):
        """Test child region runs multi-step workflow to completion."""
        workflow_region = Region(name="workflow", initial="step1", rules=[
            Rule(event_type="step1_done", target="step2"),
            Rule(event_type="step2_done", target="complete"),
        ])
        workflow_region["step1"] = Step1State()
        workflow_region["step2"] = Step2State()
        workflow_region["complete"] = WorkflowComplete()

        composite = CompositeState(
            name="workflow_composite",
            regions=ModuleList(items=[workflow_region])
        )

        main_region = Region(name="main", initial="composite", rules=[])
        main_region["composite"] = composite

        chart = StateChart(name="sequential_child", regions=[main_region])
        await chart.start()

        await asyncio.sleep(0.1)

        assert workflow_region.current_state_name == "complete"

        await chart.stop()


# ============================================================================
# Test 3: Parallel Data Collection
# ============================================================================

class CollectDataAState(State):
    """Collects data set A."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("collected")
        return {"data_a": [1, 2, 3]}


class CollectDataBState(State):
    """Collects data set B."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.03)
        await post.aforward("collected")
        return {"data_b": [4, 5, 6]}


class CollectDataCState(State):
    """Collects data set C."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("collected")
        return {"data_c": [7, 8, 9]}


class CollectionDone(FinalState):
    """Collection complete."""
    pass


class TestParallelDataCollection:
    """Test multiple regions collecting data independently."""

    @pytest.mark.asyncio
    async def test_parallel_regions_collect_independent_data(self):
        """Test each region collects data in its own context."""
        region_a = Region(name="collector_a", initial="collect", rules=[
            Rule(event_type="collected", target="done"),
        ])
        region_a["collect"] = CollectDataAState()
        region_a["done"] = CollectionDone()

        region_b = Region(name="collector_b", initial="collect", rules=[
            Rule(event_type="collected", target="done"),
        ])
        region_b["collect"] = CollectDataBState()
        region_b["done"] = CollectionDone()

        region_c = Region(name="collector_c", initial="collect", rules=[
            Rule(event_type="collected", target="done"),
        ])
        region_c["collect"] = CollectDataCState()
        region_c["done"] = CollectionDone()

        composite = CompositeState(
            name="parallel_collectors",
            regions=ModuleList(items=[region_a, region_b, region_c])
        )

        main_region = Region(name="main", initial="composite", rules=[
            Rule(event_type="all_done", target="SUCCESS"),
        ])
        main_region["composite"] = composite

        chart = StateChart(name="data_collection", regions=[main_region])

        scope = chart._scope
        main_ctx = scope.ctx(0)  # Main region context
        composite_ctx = main_ctx.child(0)  # Composite state context

        await chart.start()
        await asyncio.sleep(0.1)

        # Check each region completed
        assert region_a.current_state_name == "done"
        assert region_b.current_state_name == "done"
        assert region_c.current_state_name == "done"

        # Check composite is done running
        assert composite._run_completed.get() == True

        # Check data is in separate child contexts
        ctx_a = composite_ctx.child(0)
        ctx_b = composite_ctx.child(1)
        ctx_c = composite_ctx.child(2)

        assert ctx_a.get("data_a") == [1, 2, 3]
        assert ctx_b.get("data_b") == [4, 5, 6]
        assert ctx_c.get("data_c") == [7, 8, 9]

        # Post event to transition composite
        chart.post("all_done")
        await asyncio.sleep(0.05)

        # Main region should have transitioned to SUCCESS
        assert main_region.current_state_name == "SUCCESS"


# ============================================================================
# Test 4: Composite Completion Waits for All Regions
# ============================================================================

class FastTaskState(State):
    """Fast task that completes quickly."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("done")
        return {"fast": True}


class SlowTaskState(State):
    """Slow task that takes longer."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.1)
        await post.aforward("done")
        return {"slow": True}


class FastDone(FinalState):
    pass


class SlowDone(FinalState):
    pass


class TestCompositeCompletion:
    """Test composite completes only when ALL regions finish."""

    @pytest.mark.asyncio
    async def test_composite_waits_for_all_regions_to_complete(self):
        """Test composite doesn't complete until slowest region finishes."""
        fast_region = Region(name="fast", initial="task", rules=[
            Rule(event_type="done", target="complete"),
        ])
        fast_region["task"] = FastTaskState()
        fast_region["complete"] = FastDone()

        slow_region = Region(name="slow", initial="task", rules=[
            Rule(event_type="done", target="complete"),
        ])
        slow_region["task"] = SlowTaskState()
        slow_region["complete"] = SlowDone()

        composite = CompositeState(
            name="mixed_speed",
            regions=ModuleList(items=[fast_region, slow_region])
        )

        main_region = Region(name="main", initial="composite", rules=[
            Rule(event_type="all_complete", target="SUCCESS"),
        ])
        main_region["composite"] = composite

        chart = StateChart(name="completion_test", regions=[main_region])
        await chart.start()

        # Check fast completes first
        await asyncio.sleep(0.03)
        assert fast_region.current_state_name == "complete"
        assert slow_region.current_state_name == "task"  # Still running
        assert composite._run_completed.get() == False  # Not all done yet

        # Wait for slow to complete
        await asyncio.sleep(0.1)
        assert slow_region.current_state_name == "complete"
        assert composite._run_completed.get() == True  # Now all done

        # Post event to transition composite
        chart.post("all_complete")
        await asyncio.sleep(0.05)

        # Main region should have transitioned
        assert main_region.current_state_name == "SUCCESS"


# ============================================================================
# Test 5: Composite in State Machine Flow
# ============================================================================

class PrepareState(State):
    """Preparation before parallel work."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("prepared")
        return {"prepared": True}


class FinalizeState(State):
    """Finalization after parallel work."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("finalized")
        return {"finalized": True}


class ParallelWork1(State):
    """Parallel work 1."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("work1_done")
        return {"work1": True}


class ParallelWork2(State):
    """Parallel work 2."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("work2_done")
        return {"work2": True}


class Work1Done(FinalState):
    pass


class Work2Done(FinalState):
    pass


class FlowComplete(FinalState):
    pass


class TestCompositeInStateMachine:
    """Test composite state as part of larger state machine."""

    @pytest.mark.asyncio
    async def test_composite_embedded_in_workflow(self):
        """Test state machine: prepare → composite(parallel) → finalize."""
        # Create parallel regions for composite
        work1_region = Region(name="work1", initial="work", rules=[
            Rule(event_type="work1_done", target="done"),
        ])
        work1_region["work"] = ParallelWork1()
        work1_region["done"] = Work1Done()

        work2_region = Region(name="work2", initial="work", rules=[
            Rule(event_type="work2_done", target="done"),
        ])
        work2_region["work"] = ParallelWork2()
        work2_region["done"] = Work2Done()

        composite = CompositeState(
            name="parallel_work",
            regions=ModuleList(items=[work1_region, work2_region])
        )

        # Main workflow: sequential with composite in middle
        main_region = Region(name="workflow", initial="prepare", rules=[
            Rule(event_type="prepared", target="parallel"),
            Rule(event_type="parallel_done", target="finalize"),
            Rule(event_type="finalized", target="complete"),
        ])
        main_region["prepare"] = PrepareState()
        main_region["parallel"] = composite
        main_region["finalize"] = FinalizeState()
        main_region["complete"] = FlowComplete()

        chart = StateChart(name="workflow_with_composite", regions=[main_region])
        await chart.start()

        # Wait for prepare to complete
        await asyncio.sleep(0.02)
        assert main_region.current_state_name == "parallel"

        # Wait for composite children to complete
        await asyncio.sleep(0.1)
        assert work1_region.current_state_name == "done"
        assert work2_region.current_state_name == "done"
        assert composite._run_completed.get() == True

        # Post event to transition from composite to finalize
        chart.post("parallel_done")
        await asyncio.sleep(0.05)
        assert main_region.current_state_name == "finalize"

        # Wait for finalize and final transition
        await asyncio.sleep(0.05)
        assert main_region.current_state_name == "complete"


# ============================================================================
# Test 6: Empty Composite (Edge Case)
# ============================================================================

class TestEmptyComposite:
    """Test composite state with no regions."""

    @pytest.mark.asyncio
    async def test_empty_composite_completes_immediately(self):
        """Test composite with zero regions completes immediately."""
        composite = CompositeState(
            name="empty",
            regions=ModuleList(items=[])
        )

        main_region = Region(name="main", initial="composite", rules=[])
        main_region["composite"] = composite

        chart = StateChart(name="empty_composite", regions=[main_region])
        await chart.start()

        await asyncio.sleep(0.05)

        # Empty composite should complete immediately
        assert composite.status == ChartStatus.SUCCESS

        await chart.stop()


# ============================================================================
# Test 7: Multiple Independent Workflows
# ============================================================================

class WorkflowAStep1(State):
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("next")
        return {"workflow_a_step": 1}


class WorkflowAStep2(State):
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("done")
        return {"workflow_a_step": 2}


class WorkflowBStep1(State):
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("next")
        return {"workflow_b_step": 1}


class WorkflowBStep2(State):
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.02)
        await post.aforward("done")
        return {"workflow_b_step": 2}


class WorkflowDone(FinalState):
    pass


class TestMultipleIndependentWorkflows:
    """Test multiple independent multi-step workflows in parallel."""

    @pytest.mark.asyncio
    async def test_two_independent_workflows_run_concurrently(self):
        """Test two complete workflows execute independently in parallel."""
        workflow_a = Region(name="workflow_a", initial="step1", rules=[
            Rule(event_type="next", target="step2"),
            Rule(event_type="done", target="complete"),
        ])
        workflow_a["step1"] = WorkflowAStep1()
        workflow_a["step2"] = WorkflowAStep2()
        workflow_a["complete"] = WorkflowDone()

        workflow_b = Region(name="workflow_b", initial="step1", rules=[
            Rule(event_type="next", target="step2"),
            Rule(event_type="done", target="complete"),
        ])
        workflow_b["step1"] = WorkflowBStep1()
        workflow_b["step2"] = WorkflowBStep2()
        workflow_b["complete"] = WorkflowDone()

        composite = CompositeState(
            name="independent_workflows",
            regions=ModuleList(items=[workflow_a, workflow_b])
        )

        main_region = Region(name="main", initial="composite", rules=[])
        main_region["composite"] = composite

        chart = StateChart(name="multi_workflow", regions=[main_region])
        await chart.start()

        await asyncio.sleep(0.1)

        # Both workflows should complete
        assert workflow_a.current_state_name == "complete"
        assert workflow_b.current_state_name == "complete"

        await chart.stop()


# ============================================================================
# Test 8: Composite State Reset
# ============================================================================

class SimpleTaskState(State):
    """Simple task."""
    async def execute(self, post, **inputs):
        await asyncio.sleep(0.01)
        await post.aforward("done")
        return {"executed": True}


class SimpleDone(FinalState):
    pass


class TestCompositeReset:
    """Test composite state can be reset after completion."""

    @pytest.mark.asyncio
    async def test_composite_resets_all_child_regions(self):
        """Test resetting composite resets all child regions."""
        region1 = Region(name="region1", initial="task", rules=[
            Rule(event_type="done", target="complete"),
        ])
        region1["task"] = SimpleTaskState()
        region1["complete"] = SimpleDone()

        region2 = Region(name="region2", initial="task", rules=[
            Rule(event_type="done", target="complete"),
        ])
        region2["task"] = SimpleTaskState()
        region2["complete"] = SimpleDone()

        composite = CompositeState(
            name="resetable",
            regions=ModuleList(items=[region1, region2])
        )

        main_region = Region(name="main", initial="composite", rules=[
            Rule(event_type="completed", target="SUCCESS"),
        ])
        main_region["composite"] = composite

        chart = StateChart(name="reset_test", regions=[main_region])
        await chart.start()

        await asyncio.sleep(0.05)

        # Child regions complete
        assert region1.current_state_name == "complete"
        assert region2.current_state_name == "complete"
        assert composite._run_completed.get() == True

        # Post event to transition composite
        chart.post("completed")
        await asyncio.sleep(0.05)

        # Main region should have transitioned
        assert main_region.current_state_name == "SUCCESS"

        # Reset composite (for testing reset functionality)
        composite.reset()
        assert composite.status == ChartStatus.WAITING
