# State Chart History Implementation Plan (REVISED)

**Date**: 2025-01-20 (Revised)
**Status**: Planning
**Feature**: UML-Compliant History Pseudostates for StateChart

---

## Overview

Implement proper UML statechart history semantics using **history pseudostates**. History is recorded automatically when exiting composites and restored explicitly when transitions target history pseudostates.

### Key UML Semantics

1. **History is per-region**: Each region can have its own history pseudostate (H or H*)
2. **Recording is automatic**: When exiting a composite, runtime automatically records which substates were active
3. **Entry is explicit**: History is only restored when a transition **targets** a history pseudostate
4. **Two types**:
   - **Shallow (H)**: Restores last active immediate substate only
   - **Deep (H*)**: Restores entire nested configuration recursively

### Key Benefits
- **UML-compliant**: Follows standard statechart semantics
- **Explicit control**: Designer chooses when to resume vs restart
- **Leverages existing infrastructure**: Uses `Region._last_active_state` tracking
- **Flexible**: Each region can have different history behavior

---

## Architecture

### History Pseudostate Classes

```python
from ._base import ChartBase

class HistoryState(PseudoState):
    """Base class for history pseudostates.

    History pseudostates are special transition targets that restore
    a region's previous configuration instead of taking the initial transition.
    """
    default_target: str  # Where to go if no history exists
    history_type: Literal["shallow", "deep"]

    def __post_init__(self):
        super().__post_init__()
        # History pseudostates are always named HISTORY_SHALLOW or HISTORY_DEEP
        # to avoid conflicts with user states

class ShallowHistoryState(HistoryState):
    """Shallow history (H) - restores immediate child state only."""
    history_type: Literal["shallow"] = "shallow"

class DeepHistoryState(HistoryState):
    """Deep history (H*) - restores full nested configuration."""
    history_type: Literal["deep"] = "deep"
```

### Components

#### Region
- **Already has** `_last_active_state`: Automatically updated on every transition
- **Already has** `restore()`: Sets `_pending_target` to a specific state
- **Already has** `can_recover()` and `recover()`: Core recovery logic
- **Add**: Built-in HISTORY_SHALLOW and HISTORY_DEEP pseudostates (optional, created on demand)
- **Modify**: `transition()` to check if target is history pseudostate and trigger recovery

#### CompositeState (Recoverable)
- **Already has** `can_recover()` and `recover()`: Propagates recovery to children
- **No changes needed**: Recovery logic already correct

---

## Recovery Flow

### High-Level Flow

```
User transition → HistoryPseudostate

Region.transition(target="HISTORY_DEEP")
├─ Detect target is HistoryState
├─ Check: can_recover()?
│  ├─ YES: Call recover(history_type)
│  │      └─ Restore to _last_active_state
│  │      └─ If deep AND target is CompositeState: recursively recover children
│  │
│  └─ NO: Transition to default_target instead
│
└─ Continue with normal state entry
```

### Example: Resuming Playback

```python
# Setup
main_region = Region(
    name="main",
    initial="idle",
    rules=[
        Rule(event_type="start_playback", target="playback"),
        Rule(event_type="open_settings", when_in="playback", target="settings"),
        Rule(event_type="close_settings", when_in="settings", target="HISTORY_DEEP"),
    ]
)

# Add history pseudostate with default
main_region["HISTORY_DEEP"] = DeepHistoryState(
    name="HISTORY_DEEP",
    default_target="idle"  # First time: go to idle
)

playback = CompositeState(name="playback", regions=[transport_region])
transport_region = Region(
    name="transport",
    initial="stopped",
    rules=[
        Rule(event_type="play", when_in="stopped", target="playing"),
        Rule(event_type="pause", when_in="playing", target="paused"),
    ]
)

# User flow
chart.post("start_playback")  # main: idle → playback
                               # transport: → stopped
chart.post("play")             # transport: stopped → playing
chart.post("open_settings")    # main: playback → settings
                               # (transport._last_active_state = "playing")

# Later...
chart.post("close_settings")   # main: settings → HISTORY_DEEP
                               # Region sees target is DeepHistoryState
                               # Calls recover("deep")
                               # main restores to "playback"
                               # playback.recover("deep") restores transport to "playing"
                               # ✓ User resumes at playing!
```

---

## Implementation Changes

### 1. Create History Pseudostate Classes

**File**: `dachi/act/_chart/_state.py`

**Add**:
```python
class HistoryState(PseudoState):
    """History pseudostate base class."""
    default_target: str
    history_type: Literal["shallow", "deep"]

class ShallowHistoryState(HistoryState):
    """Shallow history (H)."""
    history_type: Literal["shallow"] = "shallow"

class DeepHistoryState(HistoryState):
    """Deep history (H*)."""
    history_type: Literal["deep"] = "deep"
```

---

### 2. Modify Region.transition() to Handle History Pseudostates

**File**: `dachi/act/_chart/_region.py`

**Changes**:

After getting `state_obj` (around line 568), add history detection:

```python
# Get target state object
try:
    state_obj: BaseState | PseudoState = self._chart_states[target]
except KeyError:
    raise RuntimeError(f"Cannot transition as State '{target}' not found in region '{self.name}'")

# NEW: Handle history pseudostates
if isinstance(state_obj, HistoryState):
    if self.can_recover():
        # Restore to last active state
        self.recover(state_obj.history_type)
        # After recover(), _pending_target is set to last active state
        # Re-call transition() to actually enter that state
        return await self.transition(post, ctx)
    else:
        # No history - use default target
        self._pending_target.set(state_obj.default_target)
        self._pending_reason.set(f"History pseudostate has no history, using default")
        return await self.transition(post, ctx)

# Rest of existing transition logic...
```

---

### 3. Remove StateChart.recovery_policy Field

**File**: `dachi/act/_chart/_chart.py`

**Remove**:
```python
recovery_policy: Literal["none", "last", "deep"] = "none"  # DELETE THIS LINE
```

The policy is now determined by which history pseudostate the transition targets, not a global setting.

---

### 4. Update Region.__post_init__ to NOT Create Built-in History States

History pseudostates are optional and created by the user when needed, not built-in like READY/SUCCESS/FAILURE/CANCELED.

**No changes needed** - users add them manually:
```python
region["HISTORY"] = ShallowHistoryState(name="HISTORY", default_target="idle")
```

---

### 5. Keep Existing Recoverable Implementation

**No changes needed** to:
- `Region.can_recover()` ✅ Already correct
- `Region.recover()` ✅ Already correct
- `CompositeState.can_recover()` ✅ Already correct
- `CompositeState.recover()` ✅ Already correct

These methods work perfectly with the history pseudostate approach.

---

### 6. Update Exports

**File**: `dachi/act/_chart/__init__.py`

**Add**:
```python
from ._state import (
    BaseState,
    State,
    StreamState,
    FinalState,
    BoundState,
    BoundStreamState,
    HistoryState,           # NEW
    ShallowHistoryState,    # NEW
    DeepHistoryState,       # NEW
)
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/act/test_region.py`

**Add test class**: `TestRegionHistory`

Tests:
- `test_history_state_with_no_history_uses_default_target`
- `test_shallow_history_restores_immediate_substate`
- `test_deep_history_restores_nested_configuration`
- `test_history_state_entry_runs_entry_actions`
- `test_history_state_first_entry_takes_default`

### E2E Tests

**File**: `tests/e2e/act/state_chart/test_statechart_history.py` (new)

**Add test class**: `TestHistoryPseudostates`

Tests:
- `test_shallow_history_in_flat_composite` - Basic H behavior
- `test_deep_history_in_nested_composite` - H* recursive restoration
- `test_history_after_explicit_exit` - Normal exit/re-entry
- `test_multiple_regions_with_separate_histories` - Independent regional history
- `test_history_to_final_state` - Edge case: history was at final
- `test_transition_to_history_vs_initial` - Explicit choice

---

## Implementation Order

1. ✅ **Region.restore()** - Already implemented
2. ✅ **Region.can_recover()** - Already implemented
3. ✅ **Region.recover()** - Already implemented
4. ✅ **CompositeState.can_recover()** - Already implemented
5. ✅ **CompositeState.recover()** - Already implemented
6. **Create HistoryState classes** in `_state.py`
7. **Modify Region.transition()** to detect and handle history pseudostates
8. **Remove StateChart.recovery_policy** field
9. **Update exports** in `__init__.py`
10. **Write unit tests** for history pseudostate behavior
11. **Write E2E tests** for composite history scenarios
12. **Run all tests** and verify integration

---

## Edge Cases

| Scenario | Behavior | Handled By |
|----------|----------|------------|
| **No history on first entry** | Takes default_target from HistoryState | `Region.transition()` checks `can_recover()` |
| **Last state no longer exists** | Warning logged, takes default_target | `Region.recover()` try/except |
| **History was at final state** | Restores to final, immediately completes | Normal final state handling |
| **Shallow history with nested active state** | Restores immediate child only, nested takes initial | `Region.recover("shallow")` doesn't recurse |
| **Deep history with nested composite** | Recursively restores all levels | `Region.recover("deep")` calls `state.recover()` |
| **Multiple regions, separate histories** | Each region has own history pseudostate | Per-region history tracking |
| **Reset clears history** | `_last_active_state` set to None | `Region.reset()` (already exists) |

---

## Design Principles

✅ **UML-compliant**: Follows standard statechart history semantics
✅ **Explicit control**: History only restores when explicitly targeted
✅ **No data duplication**: Uses existing `_last_active_state` tracking
✅ **Separation of concerns**: History logic isolated in transition handling
✅ **Backward compatible**: No breaking changes (history is opt-in)
✅ **Validation**: `can_recover()` checked before `recover()`
✅ **Per-region**: Each region manages its own history independently
✅ **Composable**: Deep history works for arbitrary nesting

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `dachi/act/_chart/_state.py` | Add HistoryState classes | +40 |
| `dachi/act/_chart/_region.py` | Modify transition() for history detection | +20 |
| `dachi/act/_chart/_chart.py` | Remove recovery_policy field | -1 |
| `dachi/act/_chart/__init__.py` | Export HistoryState classes | +3 |
| `tests/act/test_region.py` | Add TestRegionHistory class | +80 |
| `tests/e2e/act/state_chart/test_statechart_history.py` | New E2E tests | +150 |

**Total**: ~290 lines added/modified

---

## Key Differences from Previous Plan

### ❌ Removed (Wrong Approach)
- Global `recovery_policy` field on StateChart
- `set_recovery_policy()` method
- Automatic recovery on StateChart.start()
- Policy propagation through method calls

### ✅ Added (Correct UML Semantics)
- HistoryState pseudostate classes
- Explicit targeting of history in transitions
- Default target when no history exists
- Shallow vs Deep determined by pseudostate type

---

## Next Steps

1. Implement HistoryState classes
2. Modify Region.transition() to handle them
3. Remove recovery_policy from StateChart
4. Write tests
5. Validate against UML statechart examples

---

## References

- UML Statechart History States: https://en.wikipedia.org/wiki/UML_state_machine#History_state
- Harel Statecharts (original): https://www.wisdom.weizmann.ac.il/~harel/SCANNED.PAPERS/Statecharts.pdf
- SCXML History specification: https://www.w3.org/TR/scxml/#history
