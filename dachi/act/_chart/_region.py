# 1st Party
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, TypedDict, Literal, Tuple
from enum import Enum, auto

# Local
from dachi.core import BaseModule, Attr, ModuleDict
from ._state import State
from ._event import Event

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class RegionStatus(Enum):
    """Status of a Region in the state chart"""
    IDLE = "idle"           # Region not started
    ACTIVE = "active"       # Region running normally  
    FINAL = "final"         # Region reached final state
    PREEMPTING = "preempting"  # Region transitioning between states


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
    status: RegionStatus  # Required
    pending_target: Optional[str]


class Region(BaseModule):
    # ----- Spec fields (serialized) -----
    name: str
    initial: str  # Initial state name
    rules: List[Rule]

    def __post_init__(self) -> None:
        super().__post_init__()
        
        # Store State instances in module hierarchy (managed by StateChart)
        self.states = ModuleDict(items={})
        
        # Track current state with just string keys (simple data in Attr)
        self._current_state = Attr(data=self.initial)
        self._last_active_state = Attr(data=None)
        self._pending_target = Attr(data=None)
        self._pending_reason = Attr(data=None)
        self._status = Attr(data=RegionStatus.IDLE)
        
        # Build efficient rule lookup table
        self._rule_lookup: Dict[Tuple, Rule] = {}
        self._build_rule_lookup()
    
    def _build_rule_lookup(self) -> None:
        """Build efficient O(1) rule lookup table"""
        for rule in self.rules:
            if rule.get("when_in"):  # State-dependent rule
                key = (rule["when_in"], rule["event_type"])
            else:  # State-independent rule
                key = (rule["event_type"],)
            self._rule_lookup[key] = rule
    @property
    def status(self) -> RegionStatus:
        """Get current region status"""
        return self._status.data
    
    @property 
    def current_state(self) -> str:
        """Get current state name"""
        return self._current_state.data
    
    def is_final(self) -> bool:
        """Check if region is in final state"""
        return self.status == RegionStatus.FINAL
    
    def decide(self, event: "Event") -> "Decision":
        """Make routing decision based on event and current state"""
        current_state = self.current_state
        event_type = event["type"]
        
        # Check state-dependent rules first (higher precedence)
        state_dependent_key = (current_state, event_type)
        if state_dependent_key in self._rule_lookup:
            rule = self._rule_lookup[state_dependent_key]
            return {"type": "immediate", "target": rule["target"]}
            
        # Fall back to state-independent rules  
        state_independent_key = (event_type,)
        if state_independent_key in self._rule_lookup:
            rule = self._rule_lookup[state_independent_key]
            return {"type": "immediate", "target": rule["target"]}
            
        # No matching rule - stay in current state
        return {"type": "stay"}



class RuleBuilder:
    # TODO: Implement fluent API for rule building
    pass


class Decision(TypedDict, total=False):
    type: Literal["stay", "preempt", "immediate"]
    target: Optional[StateRef]

