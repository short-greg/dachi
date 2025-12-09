# 1st Party
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, TypedDict, Literal, Tuple, Set
from dataclasses import dataclass, field
import typing as t
import asyncio
import logging

import pydantic
from pydantic import Field

# Local
from dachi.core import Runtime, ModuleDict, ModuleList, PrivateRuntime
from dachi.act.comm import Ctx
from ._state import State, BaseState, PseudoState, ReadyState, FinalState, HistoryState, BASE_STATE
from ._event import Event, EventPost, ChartEventHandler
from ._base import ChartBase, ChartStatus, InvalidTransition, Recoverable

logger = logging.getLogger("dachi.statechart")

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@dataclass
class ValidationIssue:
    """A single validation issue."""
    message: str
    related_states: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


@dataclass
class ValidationResult:
    """Result of graph validation."""
    region_name: str
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def raise_if_invalid(self) -> None:
        """Raise RegionValidationError if validation failed (has errors)."""
        if not self.is_valid():
            raise RegionValidationError(self._format_issues())

    def _format_issues(self) -> str:
        """Format errors and warnings for display."""
        parts = []

        if self.errors:
            parts.append(f"Validation failed for region '{self.region_name}':")
            parts.append("Errors:")
            for i, error in enumerate(self.errors, 1):
                parts.append(f"  {i}. {error.message}")

        if self.warnings:
            if not self.errors:
                parts.append(f"Region '{self.region_name}' validation warnings:")
            parts.append("Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                parts.append(f"  {i}. {warning.message}")

        return "\n".join(parts)

    def __str__(self) -> str:
        if self.is_valid() and not self.has_warnings():
            return f"Region '{self.region_name}' validation: PASSED"
        elif self.is_valid():
            return f"Region '{self.region_name}' validation: PASSED with warnings\n{self._format_issues()}"
        return self._format_issues()


class RegionValidationError(Exception):
    """Raised when region validation fails."""
    pass


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


class Region(ChartBase, ChartEventHandler, Recoverable, t.Generic[BASE_STATE]):
    # ----- Spec fields (serialized) -----
    name: str
    initial: str  # Initial state name
    rules: List[Rule]
    states: ModuleDict[BASE_STATE] = Field(default_factory=ModuleDict)
    _cur_task: Optional[asyncio.Task] = pydantic.PrivateAttr(default=None)
    _current_state: Runtime[Optional[str]] = PrivateRuntime(default_factory=lambda: None)
    _last_active_state: Runtime[Optional[str]] = PrivateRuntime(default_factory=lambda: None)
    _pending_target: Runtime[Optional[str]] = PrivateRuntime(default_factory=lambda: None)
    _pending_reason: Runtime[Optional[str]] = PrivateRuntime(default_factory=lambda: None)
    _status: Runtime[ChartStatus] = PrivateRuntime(default_factory=lambda: ChartStatus.WAITING)
    _finished: Runtime[bool] = PrivateRuntime(default_factory=lambda: False)
    _started: Runtime[bool] = PrivateRuntime(default_factory=lambda: False)
    _stopped: Runtime[bool] = PrivateRuntime(default_factory=lambda: False)
    _stopping: Runtime[bool] = PrivateRuntime(default_factory=lambda: False)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        # Store State instances in module hierarchy
        self._state_idx_map = {}

        # Create and register built-in states (cannot be overridden by user)
        self.states["READY"] = ReadyState(name="READY")
        self.states["SUCCESS"] = FinalState(name="SUCCESS")
        self.states["FAILURE"] = FinalState(name="FAILURE")
        self.states["CANCELED"] = FinalState(name="CANCELED")

        # Track current state with just string keys (simple data in Attr)
        # Initialize to READY (always start here before start() is called)
        self._current_state = Runtime[str | None](data="READY")
        self._last_active_state = Runtime[str | None](data=None)
        self._pending_target = Runtime[str | None](data=None)
        self._pending_reason = Runtime[str | None](data=None)
        self._status = Runtime[ChartStatus](data=ChartStatus.WAITING)
        self._finished = Runtime[bool](data=False)
        self._started = Runtime[bool](data=False)
        self._stopped = Runtime[bool](data=False)
        self._stopping = Runtime[bool](data=False)

        # Build efficient rule lookup table
        self._rule_lookup: Dict[Tuple, Rule] = {}
        self._build_rule_lookup()
        # Current asyncio Task for state activity
        self._cur_task = None
    
    @pydantic.field_validator('states', mode='before')
    def validate_regions(cls, v):
        """Validate and convert regions to ModuleList

        Args:
            v: The regions input (list, ModuleList)

        Returns:
            ModuleList[BASE_STATE]: The regions as a ModuleList
        """
        # Accept any ModuleList regardless of type parameter
        # Accept ModuleList and convert

        # get the annotation args for the generic for ModuleList 
        
        base_state = cls.model_fields['states'].annotation.__pydantic_generic_metadata__['args'][0]

        if isinstance(v, list):
            converted = ModuleList[base_state](vals=v)
            return converted
        if isinstance(v, ModuleList):
            converted = ModuleList[base_state](vals=v.vals)
            return converted

        return v

    def _build_rule_lookup(self) -> None:
        """Build efficient O(1) rule lookup table"""
        for rule in self.rules:
            if rule.get("when_in"):  # State-dependent rule
                key = (rule["when_in"], rule["event_type"])
            else:  # State-independent rule
                key = (rule["event_type"],)
            self._rule_lookup[key] = rule

    def _check_reachability(self) -> List[str]:
        """Find unreachable states using BFS from initial state.

        Returns:
            List of unreachable state names (sorted)
        """
        reachable: Set[str] = {self.initial}
        queue: List[str] = [self.initial]

        while queue:
            current = queue.pop(0)

            for rule in self.rules:
                target = rule["target"]

                # Skip if already processed or not a real state
                if target in reachable or target not in self.states:
                    continue

                # Can transition from current state?
                if rule.get("when_in") == current or not rule.get("when_in"):
                    reachable.add(target)
                    queue.append(target)

        # Exclude built-in states from validation
        built_in = {'READY', 'SUCCESS', 'FAILURE', 'CANCELED'}
        all_user_states = set(self.states.keys()) - built_in

        return sorted(all_user_states - reachable)

    def _check_termination(self) -> List[str]:
        """Find states with no path to final using backward propagation.

        Note: This checks if states can reach final through the state machine's
        own transition rules. External stop() is not considered since it's an
        external interruption, not part of the state machine logic.

        Returns:
            List of non-terminating state names (sorted)
        """
        final_states = {'SUCCESS', 'FAILURE', 'CANCELED'}
        can_terminate: Set[str] = set(final_states)

        # Quick check: state-independent rule to final means ALL states can terminate
        for rule in self.rules:
            if not rule.get("when_in") and rule["target"] in final_states:
                return []  # All states have escape path

        # Backward propagation from final states
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule["target"] in can_terminate:
                    # State-independent rule: ALL states can reach target (which reaches final)
                    if not rule.get("when_in"):
                        for state_name in self.states:
                            if state_name not in can_terminate:
                                can_terminate.add(state_name)
                                changed = True
                    # State-dependent rule: only source state can reach final
                    else:
                        source = rule["when_in"]
                        if source not in can_terminate:
                            can_terminate.add(source)
                            changed = True

        # Exclude built-in states
        built_in = {'READY', 'SUCCESS', 'FAILURE', 'CANCELED'}
        all_user_states = set(self.states.keys()) - built_in

        return sorted(all_user_states - can_terminate)

    def validate(self) -> ValidationResult:
        """Validate state graph properties.

        Checks:
        1. All states are reachable from initial state (ERROR if not)
        2. All states have a path to a FinalState (WARNING if not)

        Returns:
            ValidationResult with errors and warnings

        Raises:
            RegionValidationError: If raise_if_invalid() is called and validation failed
        """
        result = ValidationResult(region_name=self.name)

        # Check reachability (ERROR)
        unreachable = self._check_reachability()
        if unreachable:
            result.errors.append(ValidationIssue(
                message=f"Unreachable states: {', '.join(unreachable)}",
                related_states=unreachable
            ))

        # Check termination (WARNING)
        non_terminating = self._check_termination()
        if non_terminating:
            result.warnings.append(ValidationIssue(
                message=f"States with no path to final: {', '.join(non_terminating)}",
                related_states=non_terminating
            ))

        return result

    def add(self, state: State) -> None:
        """Add a State instance to the region

        Args:
            state (State): The state instance to add
        """
        self.states[state.name] = state
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
        return self.states[state_name]

    def __setitem__(self, state_name: str, state: BaseState) -> None:
        """Set state by name. Calls add() internally to maintain indices.

        Args:
            state_name: Name to assign to the state
            state: The state instance to add

        Raises:
            ValueError: If state_name is a reserved name (READY, SUCCESS, FAILURE, CANCELED)
        """
        if state_name in ("READY", "SUCCESS", "FAILURE", "CANCELED"):
            raise ValueError(
                f"'{state_name}' is a reserved state name. "
                f"Use region.{state_name} to access the built-in state."
            )
        state.name = state_name
        self.add(state)

    @property
    def status(self) -> ChartStatus:
        """Get current region status"""
        return self._status.get()
    
    # @property
    # def current_state(self) -> str:
    #     """Get current state name"""
    #     return self._current_state.get()

    @property
    def READY(self) -> ReadyState:
        """Built-in ready state. Region begins here before starting."""
        return self.states["READY"]

    @property
    def SUCCESS(self) -> FinalState:
        """Built-in success state. Use for successful completion."""
        return self.states["SUCCESS"]

    @property
    def FAILURE(self) -> FinalState:
        """Built-in failure state. Reached on exception or explicit transition."""
        return self.states["FAILURE"]

    @property
    def CANCELED(self) -> FinalState:
        """Built-in canceled state. Reached on preemption or explicit transition."""
        return self.states["CANCELED"]

    def is_final(self) -> bool:
        """Check if region is in any final state (SUCCESS, FAILURE, or CANCELED)"""
        return self._current_state.get() in ("SUCCESS", "FAILURE", "CANCELED")
    
    def can_start(self) -> bool:
        """Check if region can be started"""
        return self._started.get() is False

    def can_stop(self) -> bool:
        """Check if region can be stopped"""
        return self._started.get() is True and not self._stopped.get()

    def can_reset(self) -> bool:
        return self._stopped.get() is True

    def restore(self, state: str) -> None:
        """Prepare region to start at specific state (for history restoration).

        Must be called BEFORE start(). Validates state exists.

        Args:
            state: State name to restore to

        Raises:
            InvalidTransition: If region already started
            ValueError: If state not found
        """
        if self._started.get():
            raise InvalidTransition(
                f"Cannot restore region '{self.name}' - already started"
            )

        if state not in self.states:
            raise ValueError(
                f"Cannot restore region '{self.name}' to unknown state '{state}'"
            )

        self._pending_target.set(state)
        self._pending_reason.set(f"Restored to {state} from history")

    async def start(self, post: "EventPost", ctx: Ctx) -> None:
        """Start the region. Transitions from READY to initial state.

        The region always starts in READY state. When start() is called,
        it automatically transitions READY → initial (following UML semantics).
        If restore() was called before start(), uses that state instead of initial.
        """
        if not self.can_start():
            raise InvalidTransition(
                f"Cannot start region '{self.name}' as it is already started or completed."
            )
        self._status.set(ChartStatus.RUNNING)
        self._started.set(True)

        # Check if restore() was called (for history restoration)
        if self._pending_target.get() is None:
            # No restoration - use initial state
            self._pending_target.set(self.initial)
            self._pending_reason.set("Auto-transition from READY to initial state")
        # else: restore() already set _pending_target, use that

        await self.transition(post, ctx)

    async def stop(self, post: EventPost, ctx: Ctx, preempt: bool=False) -> None:
        """Stop the region and its current state activity

        When preempt=True, requests termination and transitions to CANCELED state.
        When preempt=False, immediately cancels the running task.
        """
        if not self.can_stop():
            raise InvalidTransition(
                f"Cannot stop region '{self.name}' as it is not running."
            )

        self._stopping.set(True)

        if preempt:
            try:
                current_state_name = self.current_state_name
                current_state_obj = self.states[current_state_name]

                # Only request termination for BaseState instances (not PseudoState)
                if isinstance(current_state_obj, BaseState):
                    current_state_obj.request_termination()

                # Set pending target to CANCELED and transition
                self._pending_target.set("CANCELED")
                self._pending_reason.set("Region stopped with preemption")
                await self.transition(post, ctx)
            except KeyError:
                raise RuntimeError(
                    f"Cannot stop region '{self.name}' as current state '{self.current_state_name}' not found."
                )
        else:
            # Immediate cancellation
            if self._cur_task:
                self._cur_task.cancel()


    async def cancel(self) -> None:
        """Cancel region and its current running state."""
        if self._status.get().is_completed():
            return

        # Cancel current task
        if self._cur_task and not self._cur_task.done():
            self._cur_task.cancel()
            try:
                await self._cur_task
            except asyncio.CancelledError:
                pass
        self._cur_task = None

        await super().cancel()

    def reset(self):
        """Reset region back to READY state.

        Returns region to initial READY state, clearing all runtime flags.
        Can only be called when region is stopped.
        """
        if not self.can_reset():
            raise InvalidTransition(
                f"Cannot reset region '{self.name}' as it is not stopped."
            )

        super().reset()

        # Cancel any running tasks
        if self._cur_task and not self._cur_task.done():
            self._cur_task.cancel()
        self._cur_task = None
        for state in self.states.values():
            if isinstance(state, BaseState):
                state.reset()

        # Reset to READY state (not None)
        self._current_state.set("READY")
        self._last_active_state.set(None)
        self._pending_target.set(None)
        self._pending_reason.set(None)
        self._stopped.set(False)
        self._started.set(False)
        self._stopping.set(False)
        self._finished.set(False)

    @property
    def current_state(self) -> BaseState | PseudoState | None:
        """Get current state instance, or None if not set"""
        state_name = self._current_state.get()
        if state_name is None:
            return None
        try:
            return self.states[state_name]
        except KeyError:
            return None
        
    @property
    def current_state_name(self) -> Optional[str]:
        """Get current state name, or None if not set"""
        return self._current_state.get()

    def can_recover(self) -> bool:
        """Check if region has a last active state to recover to."""
        return self._last_active_state.get() is not None

    def recover(self, policy: Literal["shallow", "deep"]) -> None:
        """Recover to last active state using the specified policy.

        Args:
            policy: "shallow" restores immediate child only, "deep" restores full nested configuration
        """
        if not self.can_recover():
            raise RuntimeError(
                f"Cannot recover region '{self.name}' - no last active state"
            )

        last_state = self._last_active_state.get()
        self.restore(last_state)

        if policy == "deep":
            try:
                state_obj = self.states.get(last_state)
                if isinstance(state_obj, Recoverable) and state_obj.can_recover():
                    state_obj.recover(policy)
            except (KeyError, AttributeError) as e:
                logger.warning(f"Cannot recover nested state '{last_state}': {e}")

    async def transition(
        self, post: "EventPost", ctx: Ctx
    ) -> None | str:
        """Handle state transitions based on pending targets.

        Transitions to the pending target state, then checks if it's a final state.
        If final, completes the region by calling finish().

        Automatically transitions to FAILURE if current state failed with exception.
        """
        # Auto-detect state failure/cancellation and transition to appropriate built-in state
        current_state_name = self.current_state_name

        current_state_obj = self.states.get(current_state_name) if current_state_name is not None else None
        if not self._pending_target.get() and isinstance(current_state_obj, BaseState):

            if current_state_obj.status == ChartStatus.FAILURE:
                self._pending_target.set("FAILURE")
                self._pending_reason.set(f"State '{current_state_name}' failed with exception")
            elif current_state_obj.status == ChartStatus.CANCELED:
                self._pending_target.set("CANCELED")
                self._pending_reason.set(f"State '{current_state_name}' was canceled")

        if not self._pending_target.get():
            return None  # No pending transition

        target = self._pending_target.data

        # Unregister finish callback from current state (if it's a BaseState, not a PseudoState)
        if self._current_state.data is not None:
            current_state_name = self._current_state.data
            try:
                current_state_obj = self.states[current_state_name]
                # Only unregister callback for BaseState instances (not PseudoState)
                if isinstance(current_state_obj, BaseState):
                    current_state_obj.unregister_finish_callback(self.transition)
            except KeyError:
                raise RuntimeError(f"Cannot transition as current state '{current_state_name}' not found in region '{self.name}'")

        self._last_active_state.set(self._current_state.data)
        self._pending_target.set(None)
        self._pending_reason.set(None)

        # Get target state object
        try:
            state_obj: BaseState | PseudoState = self.states[target]
        except KeyError:
            raise RuntimeError(f"Cannot transition as State '{target}' not found in region '{self.name}'")

        # Handle history pseudostates
        if isinstance(state_obj, HistoryState):
            if self.can_recover():
                self.recover(state_obj.history_type)
                return await self.transition(post, ctx)
            else:
                self._pending_target.set(state_obj.default_target)
                self._pending_reason.set(f"History pseudostate has no history, using default")
                return await self.transition(post, ctx)

        # Reset state if re-entering (was previously completed)
        if isinstance(state_obj, BaseState) and state_obj.status.is_completed():
            state_obj.reset()

        # Enter new state
        self._current_state.set(target)

        # Check if new state is final
        if isinstance(state_obj, FinalState):
            # Determine status based on which final state
            if target == "SUCCESS":
                self._status.set(ChartStatus.SUCCESS)
            elif target == "FAILURE":
                self._status.set(ChartStatus.FAILURE)
            elif target == "CANCELED":
                self._status.set(ChartStatus.CANCELED)
            else:
                # Custom FinalState - use its status field
                self._status.set(state_obj.status)

            # Complete the region
            self._stopped.set(True)
            self._stopping.set(False)
            self._finished.set(True)
            await self.finish(post, ctx)
        else:
            child_post = post.sibling(target)
            child_ctx = ctx.child(self._state_idx_map[target])

            # Register transition as callback (it will be called when state finishes)
            state_obj.register_finish_callback(self.transition, post, ctx)

            # Run non-final state
            state_obj.enter(child_post, child_ctx)
            self._status.set(ChartStatus.RUNNING)
            self._cur_task = asyncio.create_task(
                state_obj.run(child_post, child_ctx)
            )

        return self._current_state.get()

    def decide(self, event: "Event") -> "Decision":
        """Make routing decision based on event and current state"""
        current_state = self.current_state_name
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
        
        return {"type": "preempt", "target": target}

    async def handle_event(
        self, event: "Event", post: EventPost, ctx: Ctx
    ) -> None:
        """Handle an incoming event and update region state accordingly"""
        if self.status != ChartStatus.RUNNING:
            return # Ignore events if not running

        cur_state = self.current_state
        if isinstance(cur_state, ChartEventHandler):
            await cur_state.handle_event(
                event,
                post.sibling(cur_state.name),
                ctx.child(self._state_idx_map[cur_state.name])
            )

        decision = self.decide(event)
        decision_type = decision["type"]

        if decision_type == "stay":
            # No state change
            return

        target = decision.get("target")
        if target is None:
            # Invalid rule - no target specified
            return

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
        if when_in is not None and when_in not in self.states:
            raise ValueError(f"Cannot add rule with when_in='{when_in}' as state not found in region '{self.name}'")
        if to_state not in self.states:
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
