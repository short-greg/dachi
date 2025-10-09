# 1st Party
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, TypedDict, Literal, Tuple, Callable
from enum import Enum
import asyncio
from ._base import ChartBase, ChartStatus, InvalidTransition

# Local
from dachi.core import Attr, ModuleDict, Ctx
from ._state import State, StreamState, BaseState
from ._event import Event, Post

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


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
        self._chart_states = ModuleDict(items={})
        self._state_idx_map = {}
        
        # Track current state with just string keys (simple data in Attr)
        self._current_state = Attr[str | None](data=None)
        self._last_active_state = Attr[str | None](data=None)
        self._pending_target = Attr[str | None](data=None)
        self._pending_reason = Attr[str | None](data=None)
        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._finished = Attr[bool](data=False)
        self._started = Attr[bool](data=False)
        self._stopped = Attr[bool](data=False)
        self._stopping = Attr[bool](data=False)

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

    def validate(self):
        """Validate region configuration"""
        pass
        # TODO: Implement validation to ensure the region's states and rules are consistent. All paths must lead to a final state.
        # if self.initial not in self._states:
        #     raise ValueError(f"Initial state '{self.initial}' not found in region '{self.name}'")
        # for rule in self.rules:
        #     if rule["target"] not in self._states:
        #         raise ValueError(f"Rule target state '{rule['target']}' not found in region '{self.name}'")
        #     if rule.get("when_in") and rule["when_in"] not in self._states:
        #         raise ValueError(f"Rule when_in state '{rule['when_in']}' not found in region '{self.name}'")

    def add(self, state: State) -> None:
        """Add a State instance to the region

        Args:
            state (State): The state instance to add
        """
        self._chart_states[state.name] = state
        self._state_idx_map[state.name] = len(self._state_idx_map)

    def __getitem__(self, state_name: str) -> BaseState:
        """Get state by name.

        Args:
            state_name: Name of the state to retrieve

        Returns:
            The state instance

        Raises:
            KeyError: If state not found
        """
        return self._chart_states[state_name]

    def __setitem__(self, state_name: str, state: BaseState) -> None:
        """Set state by name. Calls add() internally to maintain indices.

        Args:
            state_name: Name to assign to the state
            state: The state instance to add
        """
        state.name = state_name
        self.add(state)

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
        return self.status.is_completed()
    
    def can_start(self) -> bool:
        """Check if region can be started"""
        return self._started.get() is False

    def can_stop(self) -> bool:
        """Check if region can be stopped"""
        return self._started.get() is True and not self._stopped.get()

    def can_reset(self) -> bool:
        return self._stopped.get() is True

    async def start(self, post: "Post", ctx: Ctx) -> None:
        """Start the region by activating the initial state
        TODO: Make it so it will pick up if current_state already set
        """
        if not self.can_start():
            raise InvalidTransition(
                f"Cannot start region '{self.name}' as it is already started or completed."
            )       
        self._status.set(ChartStatus.RUNNING)
        self._started.set(True)
        self._pending_target.set(self.initial)
        self._pending_reason.set("Starting region")
        await self.transition(post, ctx)

    async def stop(self, post: Post, ctx: Ctx, preempt: bool=False) -> None:
        """Stop the region and its current state activity"""
        if not self.can_stop():
            raise InvalidTransition(
                f"Cannot stop region '{self.name}' as it is not running."
            )

        self._stopping.set(True)

        if preempt:
            try:
                current_state_name = self.current_state
                current_state_obj = self._chart_states[current_state_name]
                current_state_obj.request_termination()
                # await current_state_obj.exit(
                #     post.sibling(current_state_name),
                #     ctx.child(
                #         self._state_idx_map[
                #             current_state_name
                #         ]
                #     )
                # )
                await self.finish_activity(current_state_name, post, ctx)
            except KeyError:
                raise RuntimeError(f"Cannot stop region '{self.name}' as current state '{self.current_state}' not found.")
        else:
            if self._cur_task:
                self._cur_task.cancel()


    def reset(self):
        if not self.can_reset():
            raise InvalidTransition(
                f"Cannot reset region '{self.name}' as it is not stopped."
            )
        
        super().reset()
        self._stopped.set(False)
        self._started.set(False)

    async def finish_activity(self, state_name: str, post: Post, ctx: Ctx) -> None:
        """Finish the specified state

        This is executed after a State completes. If the state is final or the region is stopping, then it marks the region as completed. Otherwise, it transitions to the next state if a pending target is set.
        """
        try:
            state_obj = self._chart_states[state_name]
        except KeyError:
            raise RuntimeError(f"Cannot finish activity as State '{state_name}' not found in region '{self.name}'")

        if state_name != self._current_state.data:
            return

        if state_obj.is_final():
            self._status.set(ChartStatus.SUCCESS)
            self._stopping.set(False)
            self._stopped.set(True)
            self._cur_task = None
            # Don't set current_state to None - stay in final state
            self._pending_target.set(None)
            self._pending_reason.set(None)
            await self.finish()
        elif self._stopping.get():
            self._stopped.set(True)
            self._last_active_state.set(self._current_state.data)
            self._status.set(ChartStatus.CANCELED)
            self._stopping.set(False)
            self._cur_task = None
            self._current_state.set(None)
            self._pending_target.set(None)
            self._pending_reason.set(None)
            await self.finish()
        else: # transition to next state if defined
            self._last_active_state.set(self._current_state.data)
            self._cur_task = None
            if not await self.transition(post, ctx):
                raise RuntimeError(f"Region '{self.name}' has no pending target to transition to after state '{state_name}' finished.")

    # TODO: Complete the transition function
    async def transition(
        self, post: "Post", ctx: Ctx
    ) -> None | str:
        """Handle state transitions based on pending targets"""
        if not self._pending_target.get():
            return None  # No pending transition

        target = self._pending_target.data
        if self._current_state.data is not None:
            current_state_name = self._current_state.data
            try:
                current_state_obj = self._chart_states[current_state_name]
                current_state_obj.unregister_finish_callback(self.finish_activity)

                # await current_state_obj.exit(
                #     post.sibling(current_state_name),
                #     ctx.child(self._state_idx_map[current_state_name])
                # )
            except KeyError:
                raise RuntimeError(f"Cannot transition as current state '{current_state_name}' not found in region '{self.name}'")
        self._last_active_state.set(self._current_state.data)
        self._status.set(ChartStatus.RUNNING)
        self._pending_target.set(None)
        self._pending_reason.set(None)
        # Optionally, could log or trigger entry actions here
        try:
            state_obj: BaseState = self._chart_states[target]

            state_obj.register_finish_callback(
                self.finish_activity, target, post, ctx
            )
            child_post = post.sibling(target)
            child_ctx = ctx.child(self._state_idx_map[target])

            self._current_state.set(target)
            state_obj.enter(child_post, child_ctx)
            self._status.set(ChartStatus.RUNNING)

            self._cur_task = asyncio.create_task(
                state_obj.run(child_post, child_ctx)
            )
        except KeyError:
            raise RuntimeError(f"Cannot transition as State '{target}' not found in region '{self.name}'")
        return self._current_state.get()

    def decide(self, event: "Event") -> "Decision":
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
            state_instance = self._chart_states[current_state]
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
        if self.status != ChartStatus.RUNNING:
            return # Ignore events if not running

        decision = self.decide(event)
        decision_type = decision["type"]

        if decision_type == "stay":
            # No state change
            return

        target = decision.get("target")
        if target is None:
            # Invalid rule - no target specified
            return

        try:
            cur_state = self._chart_states[self.current_state]
        except KeyError:
            cur_state = None
        if cur_state and cur_state.run_completed():
            # Current state already completed - can
            # transition right away
            self._pending_target.set(target)
            self._pending_reason.set(event["type"])
            cur_state.exit(post.sibling(cur_state.name), ctx.child(self._state_idx_map[cur_state.name]))
            # await self.transition(post, ctx)
            # return
        elif decision_type in ("immediate", "preempt"):
            # Immediate transition to target state
            self._pending_target.set(target)
            self._pending_reason.set(event["type"])
            self._status.set(ChartStatus.PREEMPTING)
            cur_state.exit(post.sibling(cur_state.name), ctx.child(self._state_idx_map[cur_state.name]))
            if decision_type == "immediate":
                if self._cur_task:
                    self._cur_task.cancel()
            # elif decision_type == "preempt":
            #     cur_state.request_termination()
    
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
        if when_in is not None and when_in not in self._chart_states:
            raise ValueError(f"Cannot add rule with when_in='{when_in}' as state not found in region '{self.name}'")
        if to_state not in self._chart_states:
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
        self._region.add_rule(on_event=self._on_event, to_state=state, when_in=self._when_in, priority=self._priority)
    
    def priority(self, level: int) -> "RuleBuilder":
        self._priority = level
        return self


class Decision(TypedDict, total=False):
    type: Literal["stay", "preempt", "immediate"]
    target: Optional[str]
