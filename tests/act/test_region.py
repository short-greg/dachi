from __future__ import annotations
import pytest

from dachi.act._chart._region import (
    Decision,
    Region, Rule,
)
from dachi.act._chart._state import State, StreamState
from dachi.act._chart._event import Event


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


class SimpleState(State):
    async def execute(self, post, **inputs):
        pass


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
            async def astream(self, post, **inputs):
                yield

        rule = Rule(event_type="cancel", target="cancelled")
        region = Region(name="test", initial="streaming", rules=[rule])
        region._states["streaming"] = SimpleStreamState()
        region._current_state.set("streaming")
        event = Event(type="cancel")

        decision = region.decide(event)

        assert decision["type"] == "preempt"
        assert decision["target"] == "cancelled"

    def test_decide_returns_immediate_for_regular_state_transition(self):
        rule = Rule(event_type="next", target="done")
        region = Region(name="test", initial="idle", rules=[rule])
        region._states["idle"] = SimpleState()
        region._current_state.set("idle")
        event = Event(type="next")

        decision = region.decide(event)

        assert decision["type"] == "immediate"
        assert decision["target"] == "done"