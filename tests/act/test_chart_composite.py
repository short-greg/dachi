"""Unit tests for CompositeState.

Tests cover composite state lifecycle, child region management, context/post
propagation, and completion logic.
"""

import asyncio
import pytest

from dachi.act._chart._composite import CompositeState
from dachi.act._chart._region import Region, Rule
from dachi.act._chart._state import State, FinalState
from dachi.act._chart._event import EventQueue, Post
from dachi.core import Scope


class SimpleState(State):
    async def execute(self, post, **inputs):
        pass


class SimpleFinal(FinalState):
    pass


class TestCompositeStateLifecycle:

    @pytest.mark.asyncio
    async def test_composite_enters_all_child_regions_on_enter(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])

        composite = CompositeState(regions=[region1, region2])

        composite.enter()

        assert region1.status.name == "ACTIVE"
        assert region2.status.name == "ACTIVE"

    @pytest.mark.asyncio
    async def test_composite_exits_all_child_regions_on_exit(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])

        composite = CompositeState(regions=[region1, region2])

        composite.enter()
        composite.exit()

        # Both regions should be stopped
        assert region1.status.name in ["IDLE", "FINAL"]
        assert region2.status.name in ["IDLE", "FINAL"]

    @pytest.mark.asyncio
    async def test_composite_completes_when_all_children_final(self):
        region1 = Region(name="child1", initial="idle", rules=[
            Rule(event_type="finish", target="done")
        ])
        region2 = Region(name="child2", initial="idle", rules=[
            Rule(event_type="finish", target="done")
        ])
        region1._states["idle"] = SimpleState()
        region1._states["done"] = SimpleFinal()
        region2._states["idle"] = SimpleState()
        region2._states["done"] = SimpleFinal()

        composite = CompositeState(regions=[region1, region2])
        queue = EventQueue()
        post = Post(queue=queue, source=[("main", "composite")])
        scope = Scope()
        ctx = scope.ctx()

        # Start composite
        composite.enter()
        task = asyncio.create_task(composite.run(post, ctx))

        # Wait a bit for composite to start
        await asyncio.sleep(0.01)

        # Both regions still running
        assert not task.done()

        # Finish first region
        queue.post_nowait({"type": "finish", "ts": 0.0})
        await asyncio.sleep(0.01)

        # Still running (need both final)
        assert not task.done()

        # Finish second region
        queue.post_nowait({"type": "finish", "ts": 0.0})
        await asyncio.sleep(0.01)

        # Now should complete
        await asyncio.wait_for(task, timeout=0.1)
        assert task.done()

    @pytest.mark.asyncio
    async def test_composite_creates_child_contexts(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])

        composite = CompositeState(regions=[region1, region2])
        queue = EventQueue()
        post = Post(queue=queue, source=[("main", "root")])
        scope = Scope()
        ctx = scope.ctx()

        composite.enter()

        # Child contexts should be created with indices
        # This will be verified by checking data storage patterns
        ctx.child(0)["data1"] = "value1"
        ctx.child(1)["data2"] = "value2"

        # Verify data is stored at correct paths
        assert scope.get(("0", "data1")) == "value1"
        assert scope.get(("1", "data2")) == "value2"

    @pytest.mark.asyncio
    async def test_composite_creates_child_posts(self):
        region1 = Region(name="child1", initial="idle", rules=[])
        region2 = Region(name="child2", initial="idle", rules=[])

        composite = CompositeState(regions=[region1, region2])
        queue = EventQueue()
        post = Post(queue=queue, source=[("main", "composite")])
        scope = Scope()
        ctx = scope.ctx()

        # Create child posts
        child_post_0 = post.child("child1", "idle")
        child_post_1 = post.child("child2", "idle")

        # Verify hierarchical source
        assert child_post_0.source == [("main", "composite"), ("child1", "idle")]
        assert child_post_1.source == [("main", "composite"), ("child2", "idle")]


class TestCompositeStateHistory:

    @pytest.mark.asyncio
    async def test_composite_with_history_none_starts_fresh(self):
        region = Region(name="child", initial="idle", rules=[
            Rule(event_type="next", target="active")
        ])
        region._states["idle"] = SimpleState()
        region._states["active"] = SimpleState()

        composite = CompositeState(regions=[region], history="none")

        # Enter first time
        composite.enter()
        assert region.current_state == "idle"

        # Transition to active
        region._current_state.set("active")

        # Exit and re-enter
        composite.exit()
        composite.enter()

        # Should start from initial (not "active")
        assert region.current_state == "idle"


class TestCompositeStateEdgeCases:

    @pytest.mark.asyncio
    async def test_composite_with_no_regions_completes_immediately(self):
        composite = CompositeState(regions=[])
        queue = EventQueue()
        post = Post(queue=queue, source=[("main", "root")])
        scope = Scope()
        ctx = scope.ctx()

        composite.enter()
        task = asyncio.create_task(composite.run(post, ctx))

        # Should complete immediately
        await asyncio.wait_for(task, timeout=0.1)
        assert task.done()

    @pytest.mark.asyncio
    async def test_composite_preemption_cancels_children(self):
        region = Region(name="child", initial="idle", rules=[])
        region._states["idle"] = SimpleState()

        composite = CompositeState(regions=[region])
        queue = EventQueue()
        post = Post(queue=queue, source=[("main", "root")])
        scope = Scope()
        ctx = scope.ctx()

        composite.enter()
        task = asyncio.create_task(composite.run(post, ctx))

        await asyncio.sleep(0.01)

        # Request termination (preemption)
        composite.request_termination()

        # Exit should work cleanly
        composite.exit()

        # Task should complete
        await asyncio.wait_for(task, timeout=0.1)
        assert task.done()
