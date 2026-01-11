# CompositeState Implementation and Testing Plan

## Document Purpose

This document provides a comprehensive test plan for `dachi/act/_chart/_composite.py` following the systematic testing methodology used successfully for the Region module. This is part of the larger StateChart implementation effort documented in `state_chart_implementation_plan.md`.

## Current Implementation Status

**File**: `dachi/act/_chart/_composite.py` (58 lines)
**Test File**: `tests/act/test_chart_composite.py` (209 lines, incomplete)
**Last Updated**: 2025-10-08

### Implementation Overview

The `CompositeState` class extends `BaseState` to support hierarchical state machines with nested regions. It manages child regions, coordinates their execution, and tracks completion.

## Code Architecture Analysis

### Class Hierarchy

```
BaseModule
  └─ ChartBase
       └─ BaseState
            └─ CompositeState
```

### CompositeState Class Structure

#### Spec Fields (Serializable)
- `regions: ModuleList[Region]` - Child regions managed by composite

#### Runtime Fields (Instance attributes)
- `_tasks: List[asyncio.Task]` - Async tasks for child region execution
- `_finished_regions: Set[str]` - Tracks which regions have completed

#### Inherited from BaseState (via `_state.py`)
**Status tracking**:
- `_status: Attr[ChartStatus]` - Current lifecycle status
- `_termination_requested: Attr[bool]` - Preemption flag
- `_run_completed: Attr[bool]` - Execution completion flag
- `_executing: Attr[bool]` - Currently executing flag
- `_entered: Attr[bool]` - Has entered flag
- `_exiting: Attr[bool]` - Currently exiting flag

**Inherited methods** (13 from BaseState + 4 from ChartBase):
- Lifecycle: `can_enter()`, `can_run()`, `can_exit()`, `enter()`, `exit()`, `reset()`
- Status: `is_final()`, `completed()`, `get_status()`
- Termination: `request_termination()`
- Callbacks: `register_finish_callback()`, `unregister_finish_callback()`, `finish()`
- Abstract: `execute()`, `run()`

### CompositeState Methods (6 total)

#### 1. `__post_init__(self) -> None`
**Purpose**: Initialize composite state runtime fields
**Implementation**:
```python
def __post_init__(self):
    super().__post_init__()
    self._tasks = []
    self._finished_regions = set()
```
**Responsibilities**:
- Call parent `BaseState.__post_init__()` to initialize status/flags
- Initialize empty task list
- Initialize empty finished regions set

#### 2. `can_run(self) -> bool`
**Purpose**: Override BaseState check - determines if composite can execute
**Current Implementation**:
```python
def can_run(self) -> bool:
    return self._status.get().is_completed()  # ISSUE: This is wrong!
```
**Expected Behavior**: Should return True when entered but not executing
**Issue**: Currently returns True when COMPLETED, which is backwards

#### 3. `execute(self, post: Post, ctx: Ctx) -> None`
**Purpose**: Start all child regions and register completion callbacks
**Implementation**:
```python
async def execute(self, post: "Post", ctx: Ctx) -> None:
    self._tasks = []
    for region in self.regions:
        self._tasks.append(
            asyncio.create_task(
            region.start(post.child(region.name, None), ctx)
        ))
        region.register_finish_callback(self.finish_region, region.name, post, ctx)
    return None
```
**Responsibilities**:
- Clear any previous tasks
- For each child region:
  - Create async task calling `region.start()` with child Post/Ctx
  - Register `finish_region()` callback with region name
- Return None (no direct output)

**Design Questions**:
- Should this use `post.child(region.name)` instead of `post.child(region.name, None)`?
- Should tasks be awaited or just tracked?

#### 4. `run(self, post: Post, ctx: Ctx) -> None`
**Purpose**: Execute composite state - orchestrate child region execution
**Implementation**:
```python
async def run(self, post: "Post", ctx: Ctx) -> None:
    if not self.can_run():
        raise RuntimeError(f"Cannot run state '{self.name}' in {self._status.get()} state")
    if len(self.regions) == 0:
        self._status.set(ChartStatus.SUCCESS)
        await self.finish()
        return
    await self.execute(post, ctx)
    self._run_completed.set(False)
```
**Responsibilities**:
- Validate can run (will fail with current `can_run()` bug)
- Handle empty regions special case (immediate completion)
- Call `execute()` to start child regions
- Set `_run_completed` to False (regions will complete via callbacks)
- **Does NOT wait** for regions to complete (callback-driven)

#### 5. `finish_region(self, region: str) -> None`
**Purpose**: Callback invoked when a child region completes
**Implementation**:
```python
async def finish_region(self, region: str) -> None:
    self._finished_regions.add(region)
    self.regions[region].unregister_finish_callback(self.finish_region)
    if len(self._finished_regions) == len(self.regions):
        # All regions have completed
        self._tasks = []
        self._status.set(ChartStatus.SUCCESS)
        self._run_completed.set(True)
        await self.finish()
```
**Responsibilities**:
- Track region as finished
- Unregister callback from that region
- Check if all regions finished:
  - Clear tasks
  - Set status to SUCCESS
  - Set run completed
  - Call `finish()` to notify parent

#### 6. Inherited Abstract Methods
**From BaseState**:
- `enter(post, ctx)` - Inherited, no override needed
- `exit(post, ctx)` - Inherited, should it cancel child tasks?

## Critical Implementation Issues

### Issue 1: `can_run()` Logic Error (HIGH PRIORITY)
**Location**: Line 45
**Current Code**:
```python
def can_run(self) -> bool:
    return self._status.get().is_completed()
```
**Problem**: Returns True when state is COMPLETED, should return True when state is ready to run
**Expected**:
```python
def can_run(self) -> bool:
    return (self._entered.get() and
            not self._executing.get() and
            not self._run_completed.get())
```
**Impact**: `run()` will always raise RuntimeError

### Issue 2: Child Post Creation
**Location**: Line 38
**Current Code**: `post.child(region.name, None)`
**Problem**: Incorrect signature - `child()` takes one string parameter
**Expected**: `post.child(region.name)`
**Reference**: See `Post.child()` in `_event.py:163-176`

### Issue 3: Missing Child Context Creation
**Location**: Line 38
**Current Code**: Passes same `ctx` to all regions
**Problem**: Each region should get indexed child context
**Expected**: `ctx.child(i)` for region index `i`
**Reference**: See `StateChart.start()` in `_chart.py:123-129`

### Issue 4: No Termination Handling
**Location**: `execute()` and `run()` methods
**Problem**: Doesn't check `_termination_requested` flag during execution
**Expected**: Should cancel child tasks when termination requested
**Impact**: Cannot preempt composite states

### Issue 5: Task Lifecycle Management
**Location**: `execute()` and `finish_region()`
**Problem**: Tasks created but never awaited or properly cancelled
**Expected**: Either:
  - Option A: Fire-and-forget with callbacks (current approach)
  - Option B: Track and cancel on exit
**Current Issues**:
  - Tasks cleared in `finish_region()` but not cancelled
  - `exit()` doesn't cancel running tasks

### Issue 6: Missing Region Index Mapping
**Location**: `execute()` loop
**Problem**: No way to map region to index for `ctx.child(i)` calls
**Expected**: Use `enumerate()` and create `_region_idx_map` like Region does
**Reference**: See `Region.__post_init__()` in `_region.py:38-60`

### Issue 7: Missing Exit Override
**Location**: No `exit()` override
**Problem**: Inherited `exit()` doesn't know to cancel child tasks
**Expected**: Override `exit()` to stop child regions before calling `super().exit()`

### Issue 8: History Policy Not Implemented
**Location**: Class definition
**Current**: `CompositeState` has no `history` field
**Expected**: Add `history: Literal["none", "shallow", "deep"] = "none"`
**Impact**: Cannot implement history-based re-entry

## Test Plan

### Testing Strategy

Following the successful methodology from Region module testing:
1. **Analyze**: Inventory all methods and their responsibilities
2. **Plan**: Create comprehensive test cases (positive, negative, edge)
3. **Write**: Implement tests method-by-method
4. **Run**: Execute tests continuously, fix immediately
5. **Verify**: Ensure 100% method coverage

### Test Organization

#### Test Class 1: `TestCompositeStateInit` (8 tests)
Test `__post_init__()` method

**Positive cases**:
1. `test_post_init_calls_parent_init` - Verify BaseState initialization
2. `test_post_init_initializes_tasks_to_empty_list` - Check `_tasks == []`
3. `test_post_init_initializes_finished_regions_to_empty_set` - Check `_finished_regions == set()`
4. `test_post_init_sets_status_to_waiting` - Verify inherited status
5. `test_post_init_with_no_regions` - Empty regions list OK
6. `test_post_init_with_single_region` - Single region OK
7. `test_post_init_with_multiple_regions` - Multiple regions OK

**Negative cases**:
8. `test_post_init_with_none_regions_raises_error` - None not allowed

#### Test Class 2: `TestCompositeStateCanRun` (6 tests)
Test `can_run()` method override

**Positive cases**:
1. `test_can_run_returns_true_when_entered_and_not_executing` - Ready state
2. `test_can_run_returns_true_when_entered_and_not_run_completed` - After enter

**Negative cases**:
3. `test_can_run_returns_false_when_waiting` - Not entered yet
4. `test_can_run_returns_false_when_executing` - Currently running
5. `test_can_run_returns_false_when_run_completed` - Already finished
6. `test_can_run_returns_false_when_completed` - After exit **CURRENT BUG: Returns True**

#### Test Class 3: `TestCompositeStateExecute` (12 tests)
Test `execute()` method

**Positive cases**:
1. `test_execute_clears_previous_tasks` - `_tasks` reset
2. `test_execute_creates_task_for_each_region` - Task count matches region count
3. `test_execute_calls_region_start_for_each_region` - Verify start() called
4. `test_execute_passes_child_post_to_regions` - Check Post hierarchy
5. `test_execute_passes_child_ctx_to_regions` - Check Ctx child paths
6. `test_execute_registers_finish_callback_for_each_region` - Verify callbacks
7. `test_execute_returns_none` - No direct output
8. `test_execute_with_single_region` - Works with 1 region

**Negative/Edge cases**:
9. `test_execute_with_no_regions` - Empty regions list
10. `test_execute_multiple_calls_clears_tasks` - Re-entrant safety
11. `test_execute_with_invalid_region` - Region missing states
12. `test_execute_task_creation_order` - Deterministic ordering

#### Test Class 4: `TestCompositeStateRun` (10 tests)
Test `run()` method

**Positive cases**:
1. `test_run_raises_error_when_cannot_run` - Validates preconditions
2. `test_run_completes_immediately_when_no_regions` - Empty case
3. `test_run_sets_status_to_success_when_no_regions` - Empty completion
4. `test_run_calls_finish_when_no_regions` - Callback triggered
5. `test_run_calls_execute_when_has_regions` - Delegates to execute
6. `test_run_sets_run_completed_to_false_after_execute` - Flag management
7. `test_run_does_not_block_waiting_for_regions` - Async/callback pattern

**Negative/Edge cases**:
8. `test_run_raises_error_when_not_entered` - Lifecycle validation
9. `test_run_raises_error_when_already_executing` - Concurrent protection
10. `test_run_raises_error_when_already_completed` - Single execution

#### Test Class 5: `TestCompositeStateFinishRegion` (15 tests)
Test `finish_region()` callback mechanism

**Positive cases**:
1. `test_finish_region_adds_region_to_finished_set` - Tracking works
2. `test_finish_region_unregisters_callback` - Cleanup
3. `test_finish_region_does_not_finish_when_partial` - Wait for all
4. `test_finish_region_clears_tasks_when_all_complete` - Task cleanup
5. `test_finish_region_sets_status_to_success_when_all_complete` - Final status
6. `test_finish_region_sets_run_completed_when_all_complete` - Flag update
7. `test_finish_region_calls_finish_when_all_complete` - Parent notification
8. `test_finish_region_with_single_region` - One-region case
9. `test_finish_region_with_multiple_regions_sequential` - Order independence
10. `test_finish_region_completion_count_correct` - All regions tracked

**Negative/Edge cases**:
11. `test_finish_region_with_unknown_region_name` - Invalid region **ISSUE: No validation**
12. `test_finish_region_called_twice_same_region` - Idempotent
13. `test_finish_region_before_run_called` - Invalid state
14. `test_finish_region_after_composite_exited` - Late callback
15. `test_finish_region_concurrent_calls` - Thread safety

#### Test Class 6: `TestCompositeStateLifecycle` (20 tests)
Test complete lifecycle integration

**Lifecycle flow tests**:
1. `test_enter_sets_base_state_flags` - Inherited behavior
2. `test_enter_run_exit_completes_successfully` - Full cycle
3. `test_reset_after_success_allows_reentry` - Reusability
4. `test_cannot_enter_twice_without_reset` - Protection
5. `test_cannot_run_without_enter` - Lifecycle enforcement
6. `test_exit_before_regions_complete_requests_termination` - Preemption
7. `test_exit_after_regions_complete_sets_success` - Normal completion

**Child region coordination**:
8. `test_child_regions_receive_indexed_contexts` - Ctx.child(0), Ctx.child(1)
9. `test_child_regions_receive_hierarchical_posts` - Post.child() nesting
10. `test_child_region_data_isolated` - Separate context paths
11. `test_all_regions_run_in_parallel` - Concurrency
12. `test_composite_waits_for_all_regions_to_finish` - Coordination

**Finish callback propagation**:
13. `test_finish_callbacks_propagate_to_parent` - Parent notified
14. `test_finish_called_after_all_regions_complete` - Timing
15. `test_finish_not_called_if_regions_incomplete` - Correctness

**Error and edge cases**:
16. `test_exception_in_child_region_propagates` - Error handling **NOT IMPLEMENTED**
17. `test_one_region_fails_others_continue` - Isolation **DESIGN QUESTION**
18. `test_termination_cancels_all_child_tasks` - Preemption **NOT IMPLEMENTED**
19. `test_multiple_enter_run_exit_cycles` - Reusability
20. `test_nested_composite_states` - Hierarchy **FUTURE**

### Total Tests: 71 tests

## Test Execution Plan

### Phase 1: Foundation (20 tests)
**Goal**: Verify initialization and basic method structure
1. Write `TestCompositeStateInit` (8 tests)
2. Write `TestCompositeStateCanRun` (6 tests)
3. Run tests - **EXPECT 1 FAILURE** (can_run bug)
4. **DO NOT FIX YET** - Document failure

**Estimated Time**: 1 hour

### Phase 2: Core Methods (22 tests)
**Goal**: Test execute() and finish_region() independently
1. Write `TestCompositeStateExecute` (12 tests)
2. Run tests - **EXPECT FAILURES** (Post.child issue, Ctx.child missing)
3. **DO NOT FIX YET** - Document failures
4. Write `TestCompositeStateFinishRegion` (15 tests)
5. Run tests - Document all failures

**Estimated Time**: 2 hours

### Phase 3: Orchestration (10 tests)
**Goal**: Test run() method
1. Write `TestCompositeStateRun` (10 tests)
2. Run tests - **EXPECT CASCADING FAILURES** (depends on can_run)
3. **DO NOT FIX YET** - Document all failures

**Estimated Time**: 1 hour

### Phase 4: Fix Implementation (All remaining tests)
**Goal**: Fix all issues discovered in testing
1. Review all test failures
2. Fix Issue 1: `can_run()` logic
3. Fix Issue 2: `Post.child()` signature
4. Fix Issue 3: Add `ctx.child(i)` with enumeration
5. Fix Issue 6: Create region index map
6. Re-run tests - verify fixes work
7. **DO NOT proceed until all 40 tests pass**

**Estimated Time**: 2-3 hours

### Phase 5: Integration (20 tests)
**Goal**: Test complete lifecycle and edge cases
1. Write `TestCompositeStateLifecycle` (20 tests)
2. Run tests - may reveal new issues
3. Fix any new issues discovered
4. Verify all 71 tests pass

**Estimated Time**: 2-3 hours

### Phase 6: Advanced Features (Future)
**Goal**: Implement missing features
1. Add termination handling (Issue 4)
2. Add task cancellation in exit (Issue 7)
3. Add history policy support (Issue 8)
4. Write tests for new features

**Estimated Time**: 3-4 hours

## Key Design Questions to Resolve

### Question 1: Task Management Strategy
**Context**: `execute()` creates tasks but doesn't await them
**Options**:
- **A**: Fire-and-forget, rely on callbacks (current)
- **B**: Store tasks, await in `run()` with `asyncio.gather()`
- **C**: Store tasks, cancel in `exit()`

**Recommendation**: Option C - matches Region pattern
- Tasks tracked for cancellation
- Callbacks handle completion
- `exit()` can preempt cleanly

### Question 2: Error Handling in Child Regions
**Context**: If one region fails, what happens?
**Options**:
- **A**: Composite fails immediately
- **B**: Wait for all regions, report all errors
- **C**: Isolation - other regions continue

**Recommendation**: Option C for now (simpler), Option B for production
- Matches parallel semantics
- Easier to test
- Can add aggregated error handling later

### Question 3: Context Child Index
**Context**: How to map regions to context indices?
**Options**:
- **A**: Use ModuleList index (0, 1, 2...)
- **B**: Use region name as index
- **C**: Create explicit index map

**Recommendation**: Option A + C
- Enumerate regions in `execute()`
- Create `_region_idx_map` like Region does
- Use integer indices for `ctx.child(i)`

### Question 4: History Policy
**Context**: History not implemented
**Options**:
- **A**: Skip for now, test basic functionality
- **B**: Add `history` field, implement "none" only
- **C**: Full shallow/deep history

**Recommendation**: Option B
- Add field for future compatibility
- Test "none" behavior only
- Defer shallow/deep to later

## Implementation Fixes Required

### Fix 1: Correct `can_run()` Logic
**File**: `_composite.py:44-45`
**Replace**:
```python
def can_run(self) -> bool:
    return self._status.get().is_completed()
```
**With**:
```python
def can_run(self) -> bool:
    return (self._entered.get() and
            not self._executing.get() and
            not self._run_completed.get())
```

### Fix 2: Add Region Index Mapping
**File**: `_composite.py:15-19` (in `__post_init__`)
**Add after line 19**:
```python
self._region_idx_map = {}
for i, region in enumerate(self.regions):
    self._region_idx_map[region.name] = i
```

### Fix 3: Correct `execute()` Child Creation
**File**: `_composite.py:32-40`
**Replace**:
```python
async def execute(self, post: "Post", ctx: Ctx) -> None:
    self._tasks = []
    for region in self.regions:
        self._tasks.append(
            asyncio.create_task(
            region.start(post.child(region.name, None), ctx)
        ))
        region.register_finish_callback(self.finish_region, region.name, post, ctx)
    return None
```
**With**:
```python
async def execute(self, post: "Post", ctx: Ctx) -> None:
    self._tasks = []
    for i, region in enumerate(self.regions):
        child_post = post.child(region.name)
        child_ctx = ctx.child(i)
        self._tasks.append(
            asyncio.create_task(
                region.start(child_post, child_ctx)
            ))
        region.register_finish_callback(self.finish_region, region.name)
    return None
```

### Fix 4: Override `exit()` to Cancel Tasks
**File**: `_composite.py` (add new method after `execute()`)
**Add**:
```python
async def exit(self, post: Post, ctx: Ctx) -> None:
    """Exit composite and cancel all child region tasks."""
    # Cancel running tasks
    for task in self._tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete cancellation
    if self._tasks:
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # Call parent exit
    await super().exit(post, ctx)
```

### Fix 5: Add History Field
**File**: `_composite.py:13` (class definition)
**Add field**:
```python
class CompositeState(BaseState):
    """Composite state containing nested regions."""
    regions: ModuleList[Region]
    history: t.Literal["none", "shallow", "deep"] = "none"
```

## Success Criteria

### Test Coverage
- ✅ 71 tests total across 6 test classes
- ✅ 100% method coverage for CompositeState
- ✅ All positive, negative, and edge cases covered
- ✅ No skipped tests (all passing)

### Implementation Quality
- ✅ All 8 critical issues fixed
- ✅ No `can_run()` logic errors
- ✅ Proper child Post/Ctx creation
- ✅ Task lifecycle managed correctly
- ✅ Region index mapping implemented
- ✅ Exit cancels child tasks

### Documentation
- ✅ All test names follow `test_<method>_<result>_<condition>` pattern
- ✅ Test classes organized by method
- ✅ Clear assertions with descriptive failure messages

## Appendix: Code References

### Post.child() Implementation
From `_event.py:163-176`:
```python
def child(self, region_name: str) -> "Post":
    """Create a child Post with extended source hierarchy for a new region."""
    return Post(
        queue=self.queue,
        source=self.source + [(region_name, None)],
        epoch=self.epoch
    )
```

### Ctx.child() Implementation
From `_scope.py` (via Scope):
```python
def child(self, index: int) -> "Ctx":
    """Create child context for indexed access."""
    # Updates path for navigation to "./index.field"
    # Does NOT create subscope - all data in same Scope
```

### StateChart.start() Region Initialization Pattern
From `_chart.py:115-130`:
```python
async def start(self) -> None:
    self._status.set(ChartStatus.RUNNING)
    self._started_at.set(self._clock.now())

    for i, region in enumerate(self.regions):
        region.register_finish_callback(
            self.finish_region, region.name
        )
        post = self._queue.child(region.name)
        ctx = self._scope.ctx(i)  # Integer index for child
        await region.start(post, ctx)
```

### Region State Index Mapping Pattern
From `_region.py:88-90`:
```python
def add(self, state: State) -> None:
    """Add a State instance to the region"""
    self._chart_states[state.name] = state
    self._state_idx_map[state.name] = len(self._state_idx_map)
```

## Next Steps

1. ✅ Create this test plan document
2. ⏳ Review critical issues with user
3. ⏳ Begin Phase 1: Write foundation tests (TestCompositeStateInit, TestCompositeStateCanRun)
4. ⏳ Continue through phases sequentially
5. ⏳ Fix implementation issues as they're discovered
6. ⏳ Verify all 71 tests pass before declaring complete

---

**Document Version**: 1.0
**Created**: 2025-10-08
**Author**: Claude (Sonnet 4.5)
**Status**: Ready for implementation
**Related Documents**:
- `dev-docs/state_chart_implementation_plan.md` - Overall StateChart plan
- `dev-docs/state_chart_implementation_plan_for_region.md` - Region testing methodology reference
