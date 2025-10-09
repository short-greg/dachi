from __future__ import annotations
import pytest
import asyncio

from dachi.act._chart._region import (
    Decision,
    Region, Rule, RuleBuilder,
)
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.act._chart._event import Event, EventQueue, Post
from dachi.act._chart._base import ChartStatus, InvalidTransition
from dachi.core import ModuleDict, Scope, Ctx


class SimpleState(State):
    async def execute(self, post, **inputs):
        pass


class SimpleStreamState(StreamState):
    async def execute(self, post, **inputs):
        yield


class TestDecision:

    def test_stay_decision_created(self):
        decision: Decision = {"type": "stay"}
        assert decision["type"] == "stay"

    def test_preempt_decision_has_target(self):
        decision: Decision = {"type": "preempt", "target": "next_state"}
        assert decision["target"] == "next_state"

    def test_immediate_decision_has_target(self):
        decision: Decision = {"type": "immediate", "target": "emergency_state"}
        assert decision["target"] == "emergency_state"


class TestRegionInit:

    def test_post_init_creates_chart_states_module_dict(self):
        region = Region(name="test", initial="idle", rules=[])
        assert isinstance(region._chart_states, ModuleDict)

    def test_post_init_creates_state_idx_map(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region._state_idx_map == {}


    def test_post_init_sets_status_to_waiting(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region.status == ChartStatus.WAITING

    def test_post_init_initializes_all_attr_fields(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region._last_active_state.get() is None
        assert region._pending_target.get() is None
        assert region._pending_reason.get() is None
        assert region._finished.get() is False
        assert region._started.get() is False
        assert region._stopped.get() is False
        assert region._stopping.get() is False

    def test_post_init_builds_rule_lookup_table(self):
        rule = Rule(event_type="go", target="active")
        region = Region(name="test", initial="idle", rules=[rule])
        assert ("go",) in region._rule_lookup

    def test_post_init_sets_cur_task_to_none(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region._cur_task is None


class TestRegionProperties:

    def test_status_returns_current_chart_status(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region.status == ChartStatus.WAITING



class TestRegionLifecycleChecks:

    def test_can_start_returns_true_when_not_started(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region.can_start() is True

    def test_can_start_returns_false_when_already_started(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        assert region.can_start() is False

    def test_can_stop_returns_true_when_started_and_not_stopped(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        region._stopped.set(False)
        assert region.can_stop() is True

    def test_can_stop_returns_false_when_not_started(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(False)
        assert region.can_stop() is False

    def test_can_stop_returns_false_when_already_stopped(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        region._stopped.set(True)
        assert region.can_stop() is False

    def test_can_reset_returns_true_when_stopped(self):
        region = Region(name="test", initial="idle", rules=[])
        region._stopped.set(True)
        assert region.can_reset() is True

    def test_can_reset_returns_false_when_not_stopped(self):
        region = Region(name="test", initial="idle", rules=[])
        region._stopped.set(False)
        assert region.can_reset() is False


class TestRegionDecide:

    def test_decide_returns_stay_when_no_rules(self):
        region = Region(name="test", initial="idle", rules=[])
        event = Event(type="unknown")

        decision = region.decide(event)

        assert decision["type"] == "stay"

    def test_decide_matches_event_type(self):
        rule = Rule(event_type="go", target="active")
        region = Region(name="test", initial="idle", rules=[rule])
        event = Event(type="go")

        decision = region.decide(event)

        assert decision["type"] != "stay"

    def test_decide_ignores_wrong_event_type(self):
        rule = Rule(event_type="go", target="active")
        region = Region(name="test", initial="idle", rules=[rule])
        event = Event(type="stop")

        decision = region.decide(event)

        assert decision["type"] == "stay"

    def test_decide_matches_when_in_correct_state(self):
        rule = Rule(event_type="advance", target="next", when_in="waiting")
        region = Region(name="test", initial="idle", rules=[rule])
        region._current_state.set("waiting")  # Set via Attr
        event = Event(type="advance")

        decision = region.decide(event)

        assert decision["type"] != "stay"

    def test_decide_returns_preempt_for_stream_state_transition(self):
        class SimpleStreamState(StreamState):
            async def execute(self, post, **inputs):
                yield

        rule = Rule(event_type="cancel", target="cancelled")
        region = Region(name="test", initial="streaming", rules=[rule])
        region._chart_states["streaming"] = SimpleStreamState()
        region._current_state.set("streaming")
        event = Event(type="cancel")

        decision = region.decide(event)

        assert decision["type"] == "preempt"
        assert decision["target"] == "cancelled"

    def test_decide_returns_immediate_for_regular_state_transition(self):
        rule = Rule(event_type="next", target="done")
        region = Region(name="test", initial="idle", rules=[rule])
        region._chart_states["idle"] = SimpleState()
        region._current_state.set("idle")
        event = Event(type="next")

        decision = region.decide(event)

        assert decision["type"] == "immediate"
        assert decision["target"] == "done"

class TestRegionStart:

    @pytest.mark.asyncio
    async def test_start_raises_error_when_cannot_start(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(InvalidTransition):
            await region.start(post, ctx)

    @pytest.mark.asyncio
    async def test_start_sets_status_to_running(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region.status == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_sets_started_to_true(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region._started.get() is True

    @pytest.mark.asyncio
    async def test_start_calls_transition(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region.current_state == "idle"
        assert region._cur_task is not None


class TestRegionStop:

    @pytest.mark.asyncio
    async def test_stop_raises_error_when_cannot_stop(self):
        region = Region(name="test", initial="idle", rules=[])
        queue = EventQueue()
        scope = Scope(name="test")
        post = queue.child("test")
        ctx = scope.ctx(0)

        with pytest.raises(InvalidTransition):
            await region.stop(post, ctx)

    @pytest.mark.asyncio
    async def test_stop_with_preempt_false_cancels_cur_task(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        mock_task = asyncio.create_task(asyncio.sleep(0.1))
        region._cur_task = mock_task
        queue = EventQueue()
        scope = Scope(name="test")
        post = queue.child("test")
        ctx = scope.ctx(0)

        await region.stop(post, ctx, preempt=False)

        await asyncio.sleep(0)
        assert mock_task.cancelled() or mock_task.done()

    @pytest.mark.asyncio
    async def test_stop_with_preempt_false_handles_none_task(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        queue = EventQueue()
        scope = Scope(name="test")
        post = queue.child("test")
        ctx = scope.ctx(0)

        await region.stop(post, ctx, preempt=False)

    @pytest.mark.asyncio
    async def test_stop_with_preempt_true_requests_termination(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        state = SimpleState(name="active")
        region.add(state)
        region._current_state.set("active")
        state._entered.set(True)
        queue = EventQueue()
        scope = Scope(name="test")
        post = queue.child("test")
        ctx = scope.ctx(0)

        await region.stop(post, ctx, preempt=True)

        assert state._termination_requested.get() is True

    @pytest.mark.asyncio
    async def test_stop_with_preempt_true_raises_when_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        region._current_state.set("missing")
        queue = EventQueue()
        scope = Scope(name="test")
        post = queue.child("test")
        ctx = scope.ctx(0)

        with pytest.raises(RuntimeError):
            await region.stop(post, ctx, preempt=True)


class TestRegionReset:

    def test_reset_raises_when_cannot_reset(self):
        region = Region(name="test", initial="idle", rules=[])
        region._stopped.set(False)

        with pytest.raises(InvalidTransition):
            region.reset()

    def test_reset_sets_stopped_to_false(self):
        region = Region(name="test", initial="idle", rules=[])
        region._stopped.set(True)

        region.reset()

        assert region._stopped.get() is False

    def test_reset_sets_started_to_false(self):
        region = Region(name="test", initial="idle", rules=[])
        region._stopped.set(True)
        region._started.set(True)

        region.reset()

        assert region._started.get() is False


class TestRegionStateManagement:

    def test_is_final_returns_true_when_status_completed(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.SUCCESS)

        assert region.is_final() is True

    def test_is_final_returns_false_for_non_completed_status(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)

        assert region.is_final() is False

    def test_add_adds_state_to_chart_states(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")

        region.add(state)

        assert region._chart_states["active"] == state

    def test_add_updates_state_idx_map(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")

        region.add(state)

        assert region._state_idx_map["active"] == 0

    def test_add_handles_multiple_states_with_correct_indices(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="active")
        state2 = SimpleState(name="done")

        region.add(state1)
        region.add(state2)

        assert region._state_idx_map["active"] == 0
        assert region._state_idx_map["done"] == 1


class TestRegionTransition:

    @pytest.mark.asyncio
    async def test_transition_returns_none_when_no_pending_target(self):
        region = Region(name="test", initial="idle", rules=[])
        region._pending_target.set(None)
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        result = await region.transition(post, ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_transition_unregisters_callback_from_previous_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)
        region._current_state.set("idle")
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()
        
        callback = region.finish_activity
        state1.register_finish_callback(callback, "idle", post, ctx)
        assert callback in state1._finish_callbacks

        await region.transition(post, ctx)

        assert callback not in state1._finish_callbacks

    @pytest.mark.asyncio
    async def test_transition_raises_when_current_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        region._current_state.set("missing")
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(RuntimeError, match="current state 'missing' not found"):
            await region.transition(post, ctx)

    @pytest.mark.asyncio
    async def test_transition_sets_last_active_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)
        region._current_state.set("idle")
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region._last_active_state.get() == "idle"

    @pytest.mark.asyncio
    async def test_transition_sets_status_to_running(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region.status == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_transition_clears_pending_target_and_reason(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        region._pending_reason.set("test event")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region._pending_target.get() is None
        assert region._pending_reason.get() is None

    @pytest.mark.asyncio
    async def test_transition_registers_finish_callback_on_new_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region.finish_activity in state._finish_callbacks

    @pytest.mark.asyncio
    async def test_transition_sets_current_state_to_target(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region.current_state == "active"

    @pytest.mark.asyncio
    async def test_transition_calls_enter_on_new_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert state._entered.get() is True

    @pytest.mark.asyncio
    async def test_transition_creates_task_and_updates_cur_task(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region._cur_task is not None
        assert isinstance(region._cur_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_transition_raises_when_target_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        region._current_state.set(None)
        region._pending_target.set("missing")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(RuntimeError, match="State 'missing' not found"):
            await region.transition(post, ctx)

    @pytest.mark.asyncio
    async def test_transition_returns_new_current_state_name(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        result = await region.transition(post, ctx)

        assert result == "active"


class TestRegionFinishActivity:

    @pytest.mark.asyncio
    async def test_finish_activity_raises_when_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(RuntimeError, match="State 'missing' not found"):
            await region.finish_activity("missing", post, ctx)

    @pytest.mark.asyncio
    async def test_finish_activity_returns_early_when_wrong_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")
        region.add(state)
        region._current_state.set("idle")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.finish_activity("active", post, ctx)

    @pytest.mark.asyncio
    async def test_finish_activity_handles_stopping_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._stopping.set(True)
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.finish_activity("idle", post, ctx)

        assert region._stopped.get() is True
        assert region.status == ChartStatus.CANCELED
        assert region._cur_task is None
        assert region._current_state.get() is None

    @pytest.mark.asyncio
    async def test_finish_activity_handles_final_state(self):
        region = Region(name="test", initial="idle", rules=[])
        final_state = FinalState(name="done")
        region.add(final_state)
        region._current_state.set("done")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.finish_activity("done", post, ctx)

        assert region._stopped.get() is True
        assert region.status == ChartStatus.SUCCESS
        assert region._cur_task is None

    @pytest.mark.asyncio
    async def test_finish_activity_calls_transition_for_normal_completion(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)
        region._current_state.set("idle")
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.finish_activity("idle", post, ctx)

        assert region.current_state == "active"

    @pytest.mark.asyncio
    async def test_finish_activity_clears_cur_task_when_stopping(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._stopping.set(True)
        region._cur_task = asyncio.create_task(asyncio.sleep(0.1))
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.finish_activity("idle", post, ctx)

        assert region._cur_task is None

    @pytest.mark.asyncio
    async def test_finish_activity_raises_when_no_pending_target(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._pending_target.set(None)
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(RuntimeError, match="no pending target"):
            await region.finish_activity("idle", post, ctx)


class TestRegionHandleEvent:

    @pytest.mark.asyncio
    async def test_handle_event_ignores_when_not_running(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.WAITING)
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

    @pytest.mark.asyncio
    async def test_handle_event_returns_on_stay_decision(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)
        event = Event(type="unknown")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

    @pytest.mark.asyncio
    async def test_handle_event_returns_when_no_target(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

    @pytest.mark.asyncio
    async def test_handle_event_calls_exit_on_completed_state(self):
        region = Region(name="test", initial="idle", rules=[])
        rule = Rule(event_type="go", target="active")
        region.rules.append(rule)
        region._build_rule_lookup()
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._status.set(ChartStatus.RUNNING)
        state._entered.set(True)
        state._status.set(ChartStatus.SUCCESS)
        state._run_completed.set(True)
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

        assert region._pending_target.get() == "active"
        assert region._pending_reason.get() == "go"

    @pytest.mark.asyncio
    async def test_handle_event_sets_pending_for_immediate(self):
        region = Region(name="test", initial="idle", rules=[])
        rule = Rule(event_type="go", target="active")
        region.rules.append(rule)
        region._build_rule_lookup()
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._status.set(ChartStatus.RUNNING)
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        state._entered.set(True)
        state._executing.set(True)
        state._status.set(ChartStatus.RUNNING)
        event = Event(type="go")

        await region.handle_event(event, post, ctx)

        assert region._pending_target.get() == "active"
        assert region._pending_reason.get() == "go"

    @pytest.mark.asyncio
    async def test_handle_event_sets_status_to_preempting(self):
        region = Region(name="test", initial="idle", rules=[])
        rule = Rule(event_type="cancel", target="cancelled")
        region.rules.append(rule)
        region._build_rule_lookup()
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._status.set(ChartStatus.RUNNING)
        state._entered.set(True)
        state._executing.set(True)
        state._status.set(ChartStatus.RUNNING)
        event = Event(type="cancel")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

        assert region.status == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_handle_event_cancels_task_for_immediate(self):
        region = Region(name="test", initial="idle", rules=[])
        rule = Rule(event_type="go", target="active")
        region.rules.append(rule)
        region._build_rule_lookup()
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._status.set(ChartStatus.RUNNING)
        state._entered.set(True)
        state._executing.set(True)
        state._status.set(ChartStatus.RUNNING)
        task = asyncio.create_task(asyncio.sleep(0.1))
        region._cur_task = task
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)
        await asyncio.sleep(0)

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_handle_event_calls_exit_for_preempt(self):
        region = Region(name="test", initial="idle", rules=[])
        rule = Rule(event_type="cancel", target="cancelled")
        region.rules.append(rule)
        region._build_rule_lookup()
        state = SimpleStreamState(name="idle")
        region.add(state)
        region._current_state.set("idle")
        region._status.set(ChartStatus.RUNNING)
        state._entered.set(True)
        state._executing.set(True)
        state._status.set(ChartStatus.RUNNING)
        event = Event(type="cancel")
        queue = EventQueue(maxsize=10)
        post = Post(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

        assert state._exiting.get() is True


class TestRegionRuleManagement:

    def test_build_rule_lookup_creates_state_dependent_keys(self):
        rule = Rule(event_type="go", target="active", when_in="idle")
        region = Region(name="test", initial="idle", rules=[rule])

        assert ("idle", "go") in region._rule_lookup

    def test_build_rule_lookup_creates_state_independent_keys(self):
        rule = Rule(event_type="go", target="active")
        region = Region(name="test", initial="idle", rules=[rule])

        assert ("go",) in region._rule_lookup

    def test_add_rule_raises_when_when_in_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])

        with pytest.raises(ValueError, match="when_in='missing'"):
            region.add_rule(on_event="go", to_state="active", when_in="missing")

    def test_add_rule_raises_when_to_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)

        with pytest.raises(ValueError, match="target='missing'"):
            region.add_rule(on_event="go", to_state="missing", when_in="idle")

    def test_add_rule_adds_rule_to_rules_list(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)

        region.add_rule(on_event="go", to_state="active")

        assert len(region.rules) == 1

    def test_add_rule_rebuilds_lookup_table(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)

        region.add_rule(on_event="go", to_state="active")

        assert ("go",) in region._rule_lookup

    def test_add_rule_handles_state_dependent_rules(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)

        region.add_rule(on_event="go", to_state="active", when_in="idle")

        assert ("idle", "go") in region._rule_lookup

    def test_on_returns_rule_builder(self):
        region = Region(name="test", initial="idle", rules=[])

        builder = region.on("go")

        assert isinstance(builder, RuleBuilder)


class TestRuleBuilder:

    def test_when_in_sets_constraint_and_returns_self(self):
        region = Region(name="test", initial="idle", rules=[])
        builder = RuleBuilder(region, "go")

        result = builder.when_in("idle")

        assert result is builder
        assert builder._when_in == "idle"

    def test_to_calls_add_rule_with_correct_params(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)
        builder = RuleBuilder(region, "go")

        builder.to("active")

        assert len(region.rules) == 1
        assert region.rules[0]["event_type"] == "go"
        assert region.rules[0]["target"] == "active"

    def test_priority_sets_level_and_returns_self(self):
        region = Region(name="test", initial="idle", rules=[])
        builder = RuleBuilder(region, "go")

        result = builder.priority(10)

        assert result is builder
        assert builder._priority == 10

    def test_fluent_chaining_works(self):
        region = Region(name="test", initial="idle", rules=[])
        state1 = SimpleState(name="idle")
        state2 = SimpleState(name="active")
        region.add(state1)
        region.add(state2)

        region.on("go").when_in("idle").priority(5).to("active")

        assert len(region.rules) == 1
        rule = region.rules[0]
        assert rule["event_type"] == "go"
        assert rule["target"] == "active"
        assert rule["when_in"] == "idle"
        assert rule["priority"] == 5
