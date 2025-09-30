# 1st Party
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Local
from dachi.core import BaseModule
from ._state import State
from ._event import Event

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
StateRef = Union[str, State]  # name or instance


@dataclass
class RegionStatus:
    name: str
    current_state: str
    is_final: bool
    quiescing: bool
    pending_target: Optional[str]


class Rule(BaseModule):
    event_type: str
    target: Optional[StateRef]
    when_in: Optional[StateRef]
    when_prev: Optional[StateRef]
    port: Optional[str]
    priority: Optional[int]
    def __post_init__(self) -> None: ...


class Region(BaseModule):
    # ----- Spec fields (serialized) -----
    name: str
    initial: StateRef
    rules: List[Rule]

    # ----- Runtime fields (non-serialized/internal) -----
    current_state: StateRef
    last_active_state: Optional[StateRef]
    quiescing: bool = False
    pending_target: Optional[StateRef] = None
    pending_reason: Optional[str] = None

    def __post_init__(self) -> None: ...
    def is_final(self) -> bool: ...
    def current_state_name(self) -> str: ...
    def last_state_name(self) -> Optional[str]: ...
    def pending_target_name(self) -> Optional[str]: ...

    # --- Routing API (builder + management) ---
    def on(self, event_type: str) -> "RuleBuilder": ...
    def add_rule(self, rule: "Rule") -> None: ...
    def remove_rule(self, rule: "Rule") -> None: ...
    def clear_rules(self) -> None: ...
    def list_rules(self) -> List["Rule"]: ...

    # --- Decision & preemption ---
    def decide(self, event: Event) -> "Decision": ...
    def begin_quiesce(self, target: StateRef, reason: str) -> None: ...
    def end_quiesce(self) -> None: ...
    def commit(self, target: Optional[StateRef]) -> None: ...

    # --- Introspection / status ---
    def to_status(self) -> RegionStatus: ...


class RuleBuilder:
    def when_in(self, state: StateRef) -> "RuleBuilder": ...
    def when_prev(self, state: StateRef) -> "RuleBuilder": ...
    def on_port(self, port: str) -> "RuleBuilder": ...
    def with_priority(self, n: int) -> "RuleBuilder": ...
    def to(self, target: StateRef) -> "Rule": ...
    def to_final(self) -> "Rule": ...
    def ignore(self) -> "Rule": ...


class Decision:
    # TODO: implement
    pass

