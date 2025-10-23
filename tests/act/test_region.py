from __future__ import annotations
import pytest
import asyncio

from dachi.act._chart._region import (
    Decision,
    Region, Rule, RuleBuilder,
    ValidationResult, ValidationIssue, RegionValidationError,
)
from dachi.act._chart._state import State, StreamState, FinalState
from dachi.act._chart._event import Event, EventQueue, EventPost
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
        assert isinstance(region.states, ModuleDict)

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
        region.states["waiting"] = SimpleState()  # Register the state
        region._current_state.set("waiting")  # Set via Attr
        event = Event(type="advance")

        decision = region.decide(event)

        assert decision["type"] != "stay"

    def test_decide_returns_preempt_for_stream_state_transition(self):
        rule = Rule(event_type="cancel", target="cancelled")
        region = Region(name="test", initial="streaming", rules=[rule])
        # Use the module-level SimpleStreamState
        region.states["streaming"] = SimpleStreamState()
        region._current_state.set("streaming")
        event = Event(type="cancel")

        decision = region.decide(event)

        assert decision["type"] == "preempt"
        assert decision["target"] == "cancelled"

    def test_decide_returns_immediate_for_regular_state_transition(self):
        """Test decide returns preempt for regular state transitions.

        Changed from 'immediate' to 'preempt' to prevent task cancellation
        before states complete their execution.
        """
        rule = Rule(event_type="next", target="done")
        region = Region(name="test", initial="idle", rules=[rule])
        region.states["idle"] = SimpleState()
        region._current_state.set("idle")
        event = Event(type="next")

        decision = region.decide(event)

        assert decision["type"] == "preempt"
        assert decision["target"] == "done"

class TestRegionStart:

    @pytest.mark.asyncio
    async def test_start_raises_error_when_cannot_start(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        with pytest.raises(InvalidTransition):
            await region.start(post, ctx)

    @pytest.mark.asyncio
    async def test_start_sets_status_to_running(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region.status == ChartStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_sets_started_to_true(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region._started.get() is True

    @pytest.mark.asyncio
    async def test_start_calls_transition(self):
        region = Region(name="test", initial="idle", rules=[])
        region.add(SimpleState(name="idle"))
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.start(post, ctx)

        assert region.current_state_name == "idle"
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

    def test_is_final_returns_true_when_in_final_state(self):
        region = Region(name="test", initial="idle", rules=[])
        # Set current_state to one of the built-in final states
        region._current_state.set("SUCCESS")

        assert region.is_final() is True

    def test_is_final_returns_false_for_non_completed_status(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)

        assert region.is_final() is False

    def test_add_adds_state_to_chart_states(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")

        region.add(state)

        assert region.states["active"] == state

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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        # transition() is now the callback (not finish_activity)
        callback = region.transition
        state1.register_finish_callback(callback, post, ctx)
        assert callback in state1._finish_callbacks

        await region.transition(post, ctx)

        assert callback not in state1._finish_callbacks

    @pytest.mark.asyncio
    async def test_transition_raises_when_current_state_not_found(self):
        region = Region(name="test", initial="idle", rules=[])
        region._current_state.set("missing")
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        # transition() is now the callback (not finish_activity)
        assert region.transition in state._finish_callbacks

    @pytest.mark.asyncio
    async def test_transition_sets_current_state_to_target(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="active")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("active")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.transition(post, ctx)

        assert region.current_state_name == "active"

    @pytest.mark.asyncio
    async def test_transition_calls_enter_on_new_state(self):
        region = Region(name="test", initial="idle", rules=[])
        state = SimpleState(name="idle")
        region.add(state)
        region._current_state.set(None)
        region._pending_target.set("idle")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        result = await region.transition(post, ctx)

        assert result == "active"


class TestRegionHandleEvent:

    @pytest.mark.asyncio
    async def test_handle_event_ignores_when_not_running(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.WAITING)
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

    @pytest.mark.asyncio
    async def test_handle_event_returns_on_stay_decision(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)
        event = Event(type="unknown")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

    @pytest.mark.asyncio
    async def test_handle_event_returns_when_no_target(self):
        region = Region(name="test", initial="idle", rules=[])
        region._status.set(ChartStatus.RUNNING)
        event = Event(type="go")
        queue = EventQueue(maxsize=10)
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
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
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)

        assert region.status == ChartStatus.PREEMPTING

    @pytest.mark.asyncio
    async def test_handle_event_cancels_task_for_immediate(self):
        """Test handle_event uses preempt (not immediate) for regular states.

        Changed behavior: Regular states now use 'preempt' decision type,
        which calls exit() but does NOT cancel the task. This allows states
        to complete naturally and write their data to context before transitioning.
        """
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
        post = EventPost(queue=queue, source=[("test", "")])
        scope = Scope(name="test")
        ctx = scope.ctx()

        await region.handle_event(event, post, ctx)
        await asyncio.sleep(0)

        # With preempt, task is NOT cancelled - exit() is called instead
        assert not task.cancelled()
        assert state._exiting.get() is True

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
        post = EventPost(queue=queue, source=[("test", "")])
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


class TestRegionValidation:
    """Tests for Region.validate() graph validation"""

    # Reachability tests (5 tests)
    def test_validate_all_states_reachable_returns_valid_result(self):
        """All states reachable via rules"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "next", "when_in": "a", "target": "b"},
            {"event_type": "done", "when_in": "b", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["b"] = SimpleState(name="b")

        result = region.validate()
        assert result.is_valid()
        assert not result.has_warnings()

    def test_validate_detects_single_orphaned_state(self):
        """State with no incoming transition"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "done", "when_in": "a", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["orphan"] = SimpleState(name="orphan")

        result = region.validate()
        assert not result.is_valid()
        assert len(result.errors) == 1
        assert "orphan" in result.errors[0].related_states

    def test_validate_detects_multiple_orphaned_states(self):
        """Multiple unreachable states"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "done", "when_in": "a", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["orphan1"] = SimpleState(name="orphan1")
        region["orphan2"] = SimpleState(name="orphan2")

        result = region.validate()
        assert not result.is_valid()
        assert len(result.errors) == 1
        assert "orphan1" in result.errors[0].related_states
        assert "orphan2" in result.errors[0].related_states

    def test_validate_state_reachable_via_indirect_path(self):
        """State reachable through multiple hops"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "next", "when_in": "a", "target": "b"},
            {"event_type": "next", "when_in": "b", "target": "c"},
            {"event_type": "done", "when_in": "c", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["b"] = SimpleState(name="b")
        region["c"] = SimpleState(name="c")

        result = region.validate()
        assert result.is_valid()

    def test_validate_state_independent_rule_makes_all_reachable(self):
        """State-independent rule provides path to all states"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "jump", "target": "b"},  # No when_in
            {"event_type": "done", "when_in": "b", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["b"] = SimpleState(name="b")

        result = region.validate()
        assert result.is_valid()

    # Termination tests (5 tests)
    def test_validate_all_states_have_path_to_final(self):
        """All states can reach a FinalState"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "next", "when_in": "a", "target": "b"},
            {"event_type": "done", "when_in": "b", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")
        region["b"] = SimpleState(name="b")

        result = region.validate()
        assert result.is_valid()
        assert not result.has_warnings()

    def test_validate_detects_infinite_loop_with_no_escape(self):
        """State that only transitions to itself"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "loop", "when_in": "a", "target": "a"}
        ])
        region["a"] = SimpleState(name="a")

        result = region.validate()
        assert result.is_valid()  # No errors
        assert result.has_warnings()
        assert len(result.warnings) == 1
        assert "a" in result.warnings[0].related_states

    def test_validate_loop_with_escape_path_is_valid(self):
        """Loop is OK if there's an escape to final"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "loop", "when_in": "a", "target": "a"},
            {"event_type": "exit", "when_in": "a", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")

        result = region.validate()
        assert result.is_valid()
        assert not result.has_warnings()

    def test_validate_cycle_with_no_escape_detected(self):
        """Cycle A→B→C→A with no exit"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "next", "when_in": "a", "target": "b"},
            {"event_type": "next", "when_in": "b", "target": "c"},
            {"event_type": "next", "when_in": "c", "target": "a"}
        ])
        region["a"] = SimpleState(name="a")
        region["b"] = SimpleState(name="b")
        region["c"] = SimpleState(name="c")

        result = region.validate()
        assert result.is_valid()  # No errors
        assert result.has_warnings()
        assert "a" in result.warnings[0].related_states
        assert "b" in result.warnings[0].related_states
        assert "c" in result.warnings[0].related_states

    def test_validate_state_independent_rule_provides_escape(self):
        """State-independent rule to final allows all states to terminate"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "loop", "when_in": "a", "target": "a"},
            {"event_type": "abort", "target": "FAILURE"}  # No when_in
        ])
        region["a"] = SimpleState(name="a")

        result = region.validate()
        assert result.is_valid()
        assert not result.has_warnings()

    # raise_if_invalid tests (2 tests)
    def test_validate_raise_if_invalid_raises_on_failure(self):
        """raise_if_invalid() raises RegionValidationError on errors"""
        region = Region(name="test", initial="a", rules=[])
        region["a"] = SimpleState(name="a")
        region["orphan"] = SimpleState(name="orphan")

        result = region.validate()
        with pytest.raises(RegionValidationError) as exc_info:
            result.raise_if_invalid()

        assert "test" in str(exc_info.value)
        assert "orphan" in str(exc_info.value)

    def test_validate_raise_if_invalid_does_not_raise_on_success(self):
        """raise_if_invalid() does not raise when valid"""
        region = Region(name="test", initial="a", rules=[
            {"event_type": "done", "when_in": "a", "target": "SUCCESS"}
        ])
        region["a"] = SimpleState(name="a")

        result = region.validate()
        result.raise_if_invalid()  # Should not raise


class TestRegionRecovery:

    def test_can_recover_returns_true_when_last_active_state_exists(self):
        region = Region(name="test", initial="idle", rules=[])
        region._last_active_state.set("active")
        assert region.can_recover() is True

    def test_can_recover_returns_false_when_no_last_active_state(self):
        region = Region(name="test", initial="idle", rules=[])
        assert region.can_recover() is False

    def test_restore_sets_pending_target_before_start(self):
        region = Region(name="test", initial="idle", rules=[])
        region["idle"] = SimpleState(name="idle")
        region.restore("idle")
        assert region._pending_target.get() == "idle"

    def test_restore_validates_state_exists(self):
        region = Region(name="test", initial="idle", rules=[])
        with pytest.raises(ValueError, match="unknown state"):
            region.restore("nonexistent")

    def test_restore_after_start_raises_error(self):
        region = Region(name="test", initial="idle", rules=[])
        region._started.set(True)
        with pytest.raises(InvalidTransition, match="already started"):
            region.restore("idle")

    @pytest.mark.asyncio
    async def test_start_uses_restored_state_instead_of_initial(self):
        region = Region(name="test", initial="idle", rules=[])
        region["idle"] = SimpleState(name="idle")
        region["active"] = SimpleState(name="active")

        region.restore("active")

        queue = EventQueue()
        scope = Scope(name="test")
        post = EventPost(queue=queue, source=[("test", "")])
        ctx = scope.ctx(0)

        await region.start(post, ctx)
        await asyncio.sleep(0.1)

        assert region.current_state_name == "active"

    def test_recover_raises_error_if_cannot_recover(self):
        region = Region(name="test", initial="idle", rules=[])
        with pytest.raises(RuntimeError, match="no last active state"):
            region.recover("shallow")

    def test_recover_calls_restore_with_last_active_state(self):
        region = Region(name="test", initial="idle", rules=[])
        region["idle"] = SimpleState(name="idle")
        region["active"] = SimpleState(name="active")
        region._last_active_state.set("active")

        region.recover("shallow")

        assert region._pending_target.get() == "active"
