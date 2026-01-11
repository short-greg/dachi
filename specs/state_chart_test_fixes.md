# StateChart Test Fixes - Analysis & Plan

## Current Status

### Unit Tests Status ✅
- `test_chart.py`: 65 passed (CORRECT implementation reference)
- `test_region.py`: 81 passed (CORRECT implementation reference)
- `test_chart_state.py`: All passed (CORRECT implementation reference)
- `test_chart_composite.py`: All passed (CORRECT implementation reference)

### Integration/E2E Tests Status ❌
- `test_chart_integration.py`: 11 FAILED (tests are wrong)
- `test_chart_e2e.py`: 10 FAILED (tests are wrong)

### WARNING Found in Unit Tests
Even passing unit tests show: `RuntimeWarning: coroutine 'StateChart.handle_event' was never awaited`
- Location: `_event.py:59` in callback mechanism
- This indicates a **REAL BUG** in implementation

## Verified Issues

### 1. EventQueue Callback Bug 🐛 IMPLEMENTATION BUG - HIGH PRIORITY

**Evidence:**
```python
# _event.py:58-59
for callback, (args, kwargs) in self._callbacks.items():
    callback(event, *args, **kwargs)  # Sync call

# _chart.py:54-55
self._queue.register_callback(self.handle_event)  # Registers async function

# _chart.py:134
async def handle_event(self, event: Event) -> None:  # Async function!
```

**Proof it's a bug:**
- Unit tests pass but show RuntimeWarning
- Callbacks are called synchronously but handle_event is async
- Events never actually get processed

**Fix Required:**
Option A: Make callback registration async-aware
Option B: Make handle_event sync (wrapper for async)
Option C: Use asyncio.create_task() in callback dispatcher

### 2. handle_event Logic Inverted 🐛 IMPLEMENTATION BUG

**Code:** `_chart.py:137`
```python
if region.status.is_completed():  # WRONG
    await region.handle_event(event)
```

**Should be:**
```python
if not region.status.is_completed():  # or: if region.status.is_running():
    await region.handle_event(event)
```

**Evidence:**
- Docstring says "dispatching it to all running regions"
- Logic sends to completed regions
- This is obviously wrong

### 3. Missing __getitem__/__setitem__ API 📝 MISSING FEATURE

**Tests expect:** `region["state_name"] = state_instance`
**Reality:** No `__getitem__` or `__setitem__` implemented

**Fix Required:**
- Add to `Region`, `CompositeState`, `StateChart`
- `__setitem__` should call `.add()` internally
- `__getitem__` should return from `_chart_states`

### 4. Test API Misuse ❌ TESTS ARE WRONG

#### Issue 4a: Wrong StreamState Method Name
**Tests use:** `async def astream(self, post, ...)`
**Should use:** `async def execute(self, post, ...)`
**Impact:** ~10 StreamState classes in integration/e2e tests

#### Issue 4b: Wrong State Registration
**Tests use:** `region._states["name"] = state`
**Should use:** `region["name"] = state` (after we add `__setitem__`)
**Impact:** ~31 registration statements

#### Issue 4c: Non-existent join() Method
**Tests expect:** `completed = await chart.join(timeout=2.0)`
**Reality:** No `join()` method exists on StateChart

**Unit tests show correct pattern:**
- Start chart: `await chart.start()`
- Wait manually: `await asyncio.sleep(...)` + poll status
- Check completion: `chart._status.get().is_completed()`

#### Issue 4d: Wrong Status Value
**Tests expect:** `ChartStatus.FINISHED`
**Reality:** Status enum has `SUCCESS`, `FAILURE`, `CANCELED` - no `FINISHED`

## Plan

### Phase 1: Fix Implementation Bugs 🐛

#### Task 1.1: Fix EventQueue Callback for Async Handlers
**File:** `dachi/act/_chart/_event.py`
**Change:** Make callback dispatcher async-aware

**Proposed solution:**
```python
# Option: Use asyncio.create_task for async callbacks
import asyncio
import inspect

def post_nowait(self, event: Event) -> None:
    self._queue.put_nowait(event)
    for callback, (args, kwargs) in self._callbacks.items():
        if inspect.iscoroutinefunction(callback):
            asyncio.create_task(callback(event, *args, **kwargs))
        else:
            callback(event, *args, **kwargs)
```

**Verification:** RuntimeWarning should disappear from unit tests

#### Task 1.2: Fix handle_event Region Filtering
**File:** `dachi/act/_chart/_chart.py:137`
**Change:**
```python
# Before:
if region.status.is_completed():

# After:
if not region.status.is_completed():
```

**Verification:** Events should dispatch to running regions

### Phase 2: Add Missing API Methods

#### Task 2.1: Add __getitem__/__setitem__ to Region
**File:** `dachi/act/_chart/_region.py`

```python
def __getitem__(self, key: str) -> "State":
    """Get state by name."""
    return self._chart_states[key]

def __setitem__(self, key: str, state: "State") -> None:
    """Add state with given name."""
    if state.name is None:
        state.name = key
    elif state.name != key:
        raise ValueError(f"State name mismatch: key='{key}' but state.name='{state.name}'")
    self.add(state)
```

#### Task 2.2: Add __getitem__/__setitem__ to CompositeState
**File:** `dachi/act/_chart/_composite.py`
Same implementation as Region

#### Task 2.3: Add __getitem__ to StateChart (for region access)
**File:** `dachi/act/_chart/_chart.py`

```python
def __getitem__(self, key: str) -> Region:
    """Get region by name."""
    for region in self.regions:
        if region.name == key:
            return region
    raise KeyError(f"No region named '{key}'")
```

### Phase 3: Rewrite Integration Tests

#### Task 3.1: Create Integration Test Patterns Document
Document correct usage patterns based on unit tests:
- State definition (execute not astream)
- State registration (region["name"] or region.add())
- Chart lifecycle (start, poll status, stop)
- Event posting
- Context flow

#### Task 3.2: Rewrite test_chart_integration.py
**Classes to rewrite:**
1. `TestMultiStateWorkflow` (3 tests)
2. `TestConcurrentRegions` (2 tests)
3. `TestPreemptionFlows` (2 tests)
4. `TestTimerIntegration` (1 test)
5. `TestEventQueueIntegration` (2 tests)
6. `TestStateLifecycle` (1 test)

**Total:** 11 tests

**Pattern for each test:**
```python
# Define states with execute() not astream()
# Register with region.add() or region["name"]
# Start chart
# Manually wait and check status (no join())
# Use ChartStatus.SUCCESS not FINISHED
# Stop chart
```

#### Task 3.3: Rewrite test_chart_e2e.py
**Classes to rewrite:**
1. `TestWizardWorkflow` (2 tests)
2. `TestRequestResponsePattern` (2 tests)
3. `TestParallelTaskCoordination` (1 test)
4. `TestBackgroundJobCancellation` (3 tests)
5. `TestComplexStateMachine` (2 tests)

**Total:** 10 tests

### Phase 4: Verification

#### Task 4.1: Run All Unit Tests
Verify no regressions and RuntimeWarning is gone

#### Task 4.2: Run All Integration Tests
Verify all 11 tests pass

#### Task 4.3: Run All E2E Tests
Verify all 10 tests pass

#### Task 4.4: Update Documentation
Update state_chart_implementation_plan.md with completion status

## Summary

**Implementation Bugs Found:** 2 (EventQueue callback, handle_event logic)
**Missing Features:** 1 (getitem/setitem API)
**Tests to Rewrite:** 21 (11 integration + 10 e2e)

**Priority:**
1. Fix EventQueue callback (affects all event processing)
2. Fix handle_event logic (core functionality)
3. Add getitem/setitem API (test convenience)
4. Rewrite integration tests
5. Rewrite e2e tests
