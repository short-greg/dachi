# 1st Party
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, TypedDict, Literal, Tuple, Callable
from enum import Enum
import asyncio
from ._base import ChartBase, ChartStatus

# Local
from dachi.core import BaseModule, Attr, ModuleDict, Ctx
from ._state import State, StreamState, BaseState, StateStatus
from ._event import Event, Post

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# class RegionStatus(Enum):
#     """Status of a Region in the state chart"""
#     IDLE = "idle"           # Region not started
#     ACTIVE = "active"       # Region running normally  
#     FINAL = "final"         # Region reached final state
#     PREEMPTING = "preempting"  # Region transitioning between states


class Rule(TypedDict, total=False):
    event_type: str  # Required
    target: str  # Required - state name
    when_in: Optional[str]  # State-dependent constraint - state name
    port: Optional[str]
    priority: int


class RegionSnapshot(TypedDict, total=False):
    """Serializable snapshot of region state"""
    name: str  # Required
    current_state: str  # Required  
    status: ChartStatus  # Required
    pending_target: Optional[str]


class Region(ChartBase):
    # ----- Spec fields (serialized) -----
    name: str
    initial: str  # Initial state name
    rules: List[Rule]

    def __post_init__(self) -> None:
        super().__post_init__()
        
        # Store State instances in module hierarchy (managed by StateChart)
        self._states = ModuleDict(items={})
        self._state_idx_map = {}
        
        # Track current state with just string keys (simple data in Attr)
        self._current_state = Attr(data=self.initial)
        self._last_active_state = Attr(data=None)
        self._pending_target = Attr(data=None)
        self._pending_reason = Attr(data=None)
        self._status = Attr(data=ChartStatus.IDLE)
        self._activity_finished = Attr(data=False)
        
        # Build efficient rule lookup table
        self._rule_lookup: Dict[Tuple, Rule] = {}
        self._build_rule_lookup()
        # Current asyncio Task for state activity
        self._cur_task = None
    
    def _build_rule_lookup(self) -> None:
        """Build efficient O(1) rule lookup table"""
        for rule in self.rules:
            if rule.get("when_in"):  # State-dependent rule
                key = (rule["when_in"], rule["event_type"])
            else:  # State-independent rule
                key = (rule["event_type"],)
            self._rule_lookup[key] = rule

    # async def finish(self) -> None:
    #     """Mark region as finished and invoke finish callbacks"""
    #     # self._status.set(ChartStatus.COMPLETED)
    #     for callback, args, kwargs in self._finish_callbacks:
    #         if asyncio.iscoroutinefunction(callback):
    #             await callback(*args, **kwargs)
    #         else:
    #             callback(*args, **kwargs)

    # def register_finish_callback(self, callback: Callable, *args, **kwargs) -> None:
    #     """Register a callback to be called when finish() is invoked."""
    #     if callback not in self._finish_callbacks:
    #         self._finish_callbacks.append((callback, args, kwargs))

    # def unregister_finish_callback(self, callback: Callable) -> None:
    #     """Unregister a finish callback."""
    #     if callback in self._finish_callbacks:
    #         self._finish_callbacks.remove(callback)

    def validate(self):
        """Validate region configuration"""
        # TODO: Implement validation to ensure the region's states and rules are consistent
        pass 

    def add(self, state: State) -> None:
        """Add a State instance to the region

        Args:
            state (State): The state instance to add
        """
        self._states[state.name] = state
        self._state_idx_map[state.name] = len(self._state_idx_map)

    @property
    def status(self) -> ChartStatus:
        """Get current region status"""
        return self._status.get()
    
    @property 
    def current_state(self) -> str:
        """Get current state name"""
        return self._current_state.get()
    
    def is_final(self) -> bool:
        """Check if region is in final state"""
        return self.status == ChartStatus.COMPLETED
    
    async def start(self, post: "Post", ctx: Ctx) -> None:
        """Start the region by activating the initial state
        TODO: Make it so it will pick up if current_state already set
        """
        self._status.set(ChartStatus.RUNNING)
        self._pending_target.set(self.initial)
        self._pending_reason.set("Starting region")
        self.transition(post, ctx)

    def stop(self) -> None:
        """Stop the region and its current state activity"""
        if self._cur_task:
            self._cur_task.cancel()
        for key, state in self._states.items():
            if state.status == StateStatus.RUNNING:
                state.request_termination()
        self._status.set(ChartStatus.IDLE)
        self._current_state.set(None)
        self._pending_target.set(None)
        self._pending_reason.set(None)

    def finish_activity(self, state_name: str, post: Post, ctx: Ctx) -> None:
        """Finish the specified state"""

        if state_name == self._current_state.data:
            self.transition(post, ctx)

    # TODO: Complete the transition function
    async def transition(self, post: "Post", ctx: Ctx) -> None | str:
        """Handle state transitions based on pending targets"""
        if not self._pending_target.get():
            return None  # No pending transition
        
        target = self._pending_target.data
        if self.current_state.data is not None:
            current_state = self.current_state.data
            post.state(
                state_name=current_state
            ).unregister_finish_callback(
                self.finish_activity
            )
        self._last_active_state.set(self._current_state.data)
        self._status.set(ChartStatus.RUNNING)
        self._pending_target.set(None)
        self._pending_reason.set(None)
        # Optionally, could log or trigger entry actions here
        try:
            state_instance: BaseState = self._states[target]

            child_post = post.state(target)
            child_post.register_finish_callback(
                self.finish_activity, target, post, ctx
            )
            child_ctx = ctx.child(self._state_idx_map[target])

            self._current_state.set(target)
            state_instance.enter(child_post, child_ctx)
            
            self._cur_task = asyncio.create_task(
                state_instance.run(child_post, child_ctx)
            )
        except KeyError:
            raise RuntimeError(f"Cannot transition as State '{target}' not found in region '{self.name}'")
        return self._current_state.get()

    async def decide(self, event: "Event") -> "Decision":
        """Make routing decision based on event and current state"""
        current_state = self.current_state
        event_type = event["type"]

        # Check state-dependent rules first (higher precedence)
        state_dependent_key = (current_state, event_type)
        rule = self._rule_lookup.get(state_dependent_key)

        # Fall back to state-independent rules if no match
        if rule is None:
            state_independent_key = (event_type,)
            rule = self._rule_lookup.get(state_independent_key)

        # No matching rule - stay in current state
        if rule is None:
            return {"type": "stay"}

        # Found a rule - determine decision type based on current state
        target = rule["target"]

        try:
            state_instance = self._states[current_state]
            if isinstance(state_instance, StreamState):
                return {"type": "preempt", "target": target}
            else:
                return {"type": "immediate", "target": target}
        except KeyError:
            # State not registered yet, default to immediate
            return {"type": "immediate", "target": target}

    async def handle_event(
        self, event: "Event", post: Post, ctx: Ctx
    ) -> None:
        """Handle an incoming event and update region state accordingly"""
        decision = await self.decide(event)
        decision_type = decision["type"]

        if decision_type == "stay":
            # No state change
            return
        
        target = decision.get("target")
        if target is None:
            # Invalid rule - no target specified
            return
        
        cur_state = self._states.get(self.current_state)
        if cur_state and cur_state.completed():
            # Current state already completed - can 
            # transition right away
            self._pending_target.set(target)
            self._pending_reason.set(event["type"])
            self.transition(post, ctx)
            return

        if decision_type in ("immediate", "preempt"):
            # Immediate transition to target state
            self._pending_target.set(target)
            self._pending_reason.set(event["type"])
            self._status.set(ChartStatus.PREEMPTING)
            if decision_type == "immediate":
                self._cur_task.cancel()
            elif decision_type == "preempt":
                cur_state.request_termination()
    
    def on(self, event_type: str) -> "RuleBuilder":
        """Begin building a rule for the specified event type"""
        return RuleBuilder(self, event_type)
    
    def add_rule(self, on_event: str, to_state: str, when_in: Optional[str] = None, priority: int = 0) -> None:
        """Add a rule to the region

        Args:
            on_event (str): The event type to trigger the rule
            target (str): The target state name
            when_in (Optional[str], optional): The state name condition. Defaults to None.
            priority (int, optional): The priority of the rule. Defaults to 0.
        """
        if when_in not in self._states:
            raise ValueError(f"Cannot add rule with when_in='{when_in}' as state not found in region '{self.name}'")
        if to_state not in self._states:
            raise ValueError(f"Cannot add rule with target='{to_state}' as state not found in region '{self.name}'")
        rule: Rule = {
            "event_type": on_event,
            "target": to_state,
            "priority": priority
        }
        if when_in:
            rule["when_in"] = when_in
        self.rules.append(rule)
        self._build_rule_lookup()


class RuleBuilder:
    # TODO: Implement fluent API for rule building
    def __init__(
        self, region: Region, on_event: str, when_in: Optional[str] = None, to_state: Optional[str] = None, priority: int = 0
    ):
        self._region = region
        self._on_event = on_event
        self._when_in = when_in
        self._to_state = to_state
        self._priority = priority

    def when_in(self, state: str) -> "RuleBuilder":
        self._when_in = state
        return self
    
    def to(self, state: str) -> "RuleBuilder":
        self._to_state = state
        self._region.add_rule(on_event=self._on_event, target=state, when_in=self._when_in, priority=self._priority)
    
    def priority(self, level: int) -> "RuleBuilder":
        self._priority = level
        return self


class Decision(TypedDict, total=False):
    type: Literal["stay", "preempt", "immediate"]
    target: Optional[str]
