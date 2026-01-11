# Process Module Refactoring Plan

## Overview
Update `dachi/proc` module after core refactoring completed (247 tests passing for dachi/core).

## Status: IN PROGRESS - Phase 4 Complete, Phase 5 Pending

---

## Phase 1: func_arg_model Testing and Fixes ✅ COMPLETED

### Goals
- Extract and test `func_arg_model` in isolation
- Ensure it correctly handles all Python parameter types
- Fix argument ordering and wrapper types

### Completed Work

#### Files Created
- `/Users/shortg/Development/dachi/dachi/proc/_arg_model.py` - Extracted argument model utilities
- `/Users/shortg/Development/dachi/tests/proc/test_process_func_arg.py` - Comprehensive test suite (36 tests)

#### Key Fixes Applied
1. **Default Factory Pattern** - Fixed Pydantic field defaults to use `Field(default_factory=...)`
2. **Lambda Variable Capture** - Fixed closure issues by using default parameters
3. **Parameter Type Handling**:
   - `POSITIONAL_ONLY` (x, /) → Regular fields (NOT wrapped)
   - `POSITIONAL_OR_KEYWORD` (x: int) → Regular fields
   - `KEYWORD_ONLY` (*, x: int) → Wrapped in `KWOnly[T]`
   - `VAR_POSITIONAL` (*args) → Wrapped in `PosArgs[T]` (list)
   - `VAR_KEYWORD` (**kwargs) → Wrapped in `KWArgs[T]` (dict)
4. **Required vs Optional** - Required keyword-only params (no default) get `...` not default factory
5. **Documentation** - Updated docstrings to reflect actual wrapper usage

#### Test Results
- ✅ 36/36 tests passing
- All parameter types covered
- Ordering tests verified
- Reference resolution tested

#### Current Understanding

**Wrapper Types:**
```python
class KWOnly[V]:
    """Wrapper for keyword-only arguments (*, x: int)"""
    data: V

class PosArgs[V]:
    """Wrapper for VAR_POSITIONAL (*args) arguments"""
    data: List[V]

class KWArgs[V]:
    """Wrapper for VAR_KEYWORD (**kwargs) arguments"""
    data: Dict[str, V]
```

**BaseArgs.get_args():**
```python
def get_args(self, by: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Returns (positional_args, keyword_args) with references resolved"""
    # Separates regular params, KWOnly, PosArgs, KWArgs
    # Resolves Ref instances using 'by' mapping
    return ([*args, *pos_args], {**kw_only, **kwargs})
```

---

## Phase 2: BaseArgs.build() Class Method ✅ COMPLETED

### Goal
Create a class method to build Args instances from runtime *args, **kwargs.

### Completed Work

#### Implementation Details
- Added `BaseArgs.build()` class method in [_arg_model.py:78-150](/Users/shortg/Development/dachi/dachi/proc/_arg_model.py#L78-L150)
- Uses `issubclass()` to detect wrapper types (KWOnly, PosArgs, KWArgs)
- Matches positional args to regular fields in order
- Wraps keyword-only args in their specific generic KWOnly subclass
- Handles excess positional args via PosArgs wrapper
- Handles excess keyword args via KWArgs wrapper
- Validates duplicate argument provision

#### Test Results
- ✅ 12/12 new build() tests passing
- ✅ 48/48 total tests passing in test_process_func_arg.py

#### Key Implementation Notes
1. **Wrapper Detection**: Used `issubclass()` instead of checking `__origin__` because Pydantic creates specialized subclasses for each generic instantiation
2. **Generic Instantiation**: Must instantiate the specific generic subclass (e.g., `KWOnly[str]`) not the base class
3. **Field Ordering**: Preserved correct positional argument order by iterating through field_items
4. **Error Handling**: Raises TypeError for duplicate arguments and unexpected parameters

### Original Requirements
```python
@classmethod
def build(cls, *args, **kwargs) -> 'BaseArgs':
    """Build an Args instance from runtime arguments.

    Takes the actual positional and keyword arguments that would be
    passed to a function and wraps them appropriately based on the
    model's field structure.

    Args:
        *args: Positional arguments to the function
        **kwargs: Keyword arguments to the function

    Returns:
        An instance of the Args model with fields populated

    Example:
        # For: def forward(self, x: int, y: str, *, z: bool = True)
        model = func_arg_model(Cls, Cls.forward)

        # Build from runtime args
        args_instance = model.build(42, "hello", z=False)

        # args_instance.x == 42
        # args_instance.y == "hello"
        # args_instance.z == KWOnly(data=False)
    """
```

### Edge Cases Handled
- ✅ More positional args than fields (with/without VAR_POSITIONAL)
- ✅ Keyword args that don't match any field (with/without VAR_KEYWORD)
- ✅ Mix of positional and keyword args for same param (raises TypeError)
- ✅ Required fields satisfied through Pydantic validation
- ✅ Default values for missing optional fields

### Tests Added
- ✅ Basic positional args
- ✅ Keyword-only args
- ✅ *args overflow (PosArgs)
- ✅ **kwargs overflow (KWArgs)
- ✅ Mixed positional and keyword
- ✅ Default value handling
- ✅ Roundtrip with get_args()
- ✅ Empty PosArgs and KWArgs

---

## Phase 3: Update _process.py ✅ COMPLETED

### Goals
- Import from `_arg_model.py` instead of duplicating code
- Update `BaseProcessCall` to use new arg model
- Ensure `depends_on()` works correctly
- Add error handling for Pydantic model generation failures

### Completed Work

#### Code Changes in [_process.py](/Users/shortg/Development/dachi/dachi/proc/_process.py)

1. **Imports Added** (lines 43-50)
   - Imported `Ref`, `KWOnly`, `PosArgs`, `KWArgs`, `BaseArgs`, `func_arg_model` from `._arg_model`
   - Removed 240+ lines of duplicated code (former lines 46-289)

2. **Error Handling in __init_subclass__** (4 locations)
   - `Process.__init_subclass__` (lines 68-90): Wrapped model generation in try/except
   - `AsyncProcess.__init_subclass__` (lines 141-163): Added error handling
   - `StreamProcess.__init_subclass__` (lines 280-302): Added error handling
   - `AsyncStreamProcess.__init_subclass__` (lines 345-367): Added error handling

   Error handling pattern:
   - Catches all exceptions during func_arg_model generation
   - Issues UserWarning with clear explanation
   - Sets all ArgModel class variables to None
   - Allows class definition to succeed

3. **Defensive Checks in forward_process_call Methods** (4 locations)
   - `Process.forward_process_call` (lines 109-134)
   - `AsyncProcess.aforward_process_call` (lines 189-214)
   - `StreamProcess.stream_process_call` (lines 335-360)
   - `AsyncStreamProcess.astream_process_call` (lines 410-435)

   Pattern:
   - Checks if ArgModel is None before use
   - Raises RuntimeError with clear message if None
   - Prevents silent failures

4. **Fixed BaseProcessCall.depends_on()** (line 191)
   - Changed from `for field in self.args.model_fields.values()` with `field.name`
   - To `for field_name in self.args.model_fields.keys()`
   - Fixes FieldInfo attribute error

#### Test Results
- ✅ 48/48 func_arg tests passing
- ✅ func_arg_model integration verified
- ⚠️ Full proc tests blocked by pre-existing _graph.py import error (Phase 4 issue)

#### Error Handling Behavior

**When Pydantic model generation succeeds:**
- ArgModel class variables populated
- ProcessCall creation works normally
- No warnings issued

**When Pydantic model generation fails:**
- UserWarning issued at class definition time
- ArgModel class variables set to None
- Direct method calls (forward/aforward/stream) still work
- ProcessCall creation raises clear RuntimeError

### Key Improvements
1. **No Code Duplication** - Eliminated 240+ lines of duplicate code
2. **Clear Error Messages** - Users know immediately if ProcessCall cannot be created
3. **Graceful Degradation** - Process classes can still be defined and used directly even if arg model generation fails
4. **Better Maintainability** - Single source of truth for arg model logic in `_arg_model.py`

---

## Phase 4: DataFlow Import Error Resolution ✅ COMPLETED

### Issues Resolved

1. **DataFlow field() usage** - Line 434 used `field()` from dataclasses instead of `pydantic.Field()`
2. **typing.Dict namespace collision** - The function `t()` at line 320 was shadowing the `typing` module import
3. **Missing import** - `_optim.py` imported non-existent `modfield` from dachi.core

### Changes Made

#### [_graph.py](/Users/shortg/Development/dachi/dachi/proc/_graph.py)

1. **Consolidated imports** (lines 38-49)
   - Removed duplicate `import typing` statements
   - Cleaned up redundant imports
   - Used `typing` directly instead of alias `t` to avoid collision

2. **Fixed DataFlow fields** (lines 421-426)
   - Changed `field(default_factory=list)` to `pydantic.Field(default_factory=list)`
   - Changed `typing.List` to `list` (Python 3.9+ syntax)
   - Changed `typing.Dict` to `dict` to avoid namespace issues

3. **Function renaming** (line 320)
   - User renamed function `t()` to `sync_t()` to avoid shadowing `typing` module

#### [_optim.py](/Users/shortg/Development/dachi/dachi/proc/_optim.py)

- Removed non-existent `modfield` from import (line 12)

### Test Results

- ✅ **dachi.proc imports successfully!**
- ✅ **48/48 func_arg tests passing** - All refactoring intact
- ✅ **210/253 total proc tests passing** (83% pass rate)
- ⚠️ **43 test failures** - Pre-existing test compatibility issues (see Session Continuity section)
- ⚠️ **1 test error** - test_graph.py imports non-existent SerialDict

### Key Achievement
**All import errors resolved!** The refactoring from Phases 1-3 is fully integrated and working.
**Framework code is functional** - test failures are from outdated test implementations, not regressions.

---

## Phase 5: Fix Remaining Test Failures ✅ COMPLETED

### Goal
Review and fix all test failures to ensure the refactoring hasn't caused regressions.

### Final Test Status
- ✅ **353/353 proc tests passing (100%)**
- ✅ All test failures resolved
- ✅ All Pydantic compatibility issues fixed

### Completed Fixes

#### test_resp.py (3 failures → 41/41 passing) ✅
**Issue**: TupleOut type validation rejecting ModuleList instances
**Root Cause**: Pydantic strict validation + wrong test design (using Process instead of ToOut)
**Fixes Applied**:
1. Added `field_validator` for `processors` field to accept ModuleList instances
2. Fixed tests to use `ToOut` subclasses (with all required abstract methods)
3. Added negative test for type validation (Process rejection)
4. Updated `StructOut` type annotation to accept `Type[BaseModel]` not just instances

#### test_instruct.py (11 failures → 11/11 passing) ✅
**Issue**: Multiple Pydantic compatibility issues in instruction decorators
**Root Causes**:
1. Private attributes (`_docstring`, `_name`, etc.) declared as regular fields
2. Decorator instantiation using positional arguments
3. `spawn()` methods not properly creating Pydantic instances
4. Wrong `to_msg` default (NullToPrompt expects Msg, but receives str)
5. Engine field not accepting string for lazy lookup

**Fixes Applied**:
1. Converted `_docstring`, `_name`, `_signature`, `_sig_params`, `_return_annotation` to `PrivateAttr()` in IBase
2. Fixed all decorator instantiations (FuncDec, AFuncDec, StreamDec, AStreamDec) to use keyword arguments
3. Fixed all 5 `spawn()` methods to properly instantiate with `engine`, `inst`, `to_msg` and then set private attrs
4. Added `kwargs` property to access `_kwargs` for backward compatibility
5. Changed default `to_msg` from `NullToPrompt()` to `ToText()` (handles string → Prompt conversion)
6. Made `engine` parameter optional and accept `str` type for attribute lookup
7. Fixed `doc` parameter to use empty string default instead of None

#### test_optim.py (6 failures → 6/6 passing) ✅
**Issue**: MockProcess test class using `__init__` to set attributes
**Root Cause**: Pydantic disallows arbitrary attribute setting in `__init__`
**Fixes Applied**:
1. Converted MockProcess to use Pydantic field declarations instead of `__init__`
2. Changed all test instantiations from `MockProcess(response_json)` to `MockProcess(response_json=response_json)`

#### test_process.py (Previously completed in earlier session) ✅
**Issue**: Test helper classes using `__init__` pattern
**Fix**: Converted to Pydantic Field declarations

#### test_process_call.py (Previously completed in earlier session) ✅
**Issue**: Direct ProcessCall instantiation instead of using API
**Fix**: Updated to use `forward_process_call()` method

#### test_graph.py (Previously completed in earlier session) ✅
**Issues**: SerialDict import error, positional arguments, type compatibility
**Fixes**: Removed SerialDict usage, added hashability, fixed async calls, updated tests

### Key Pydantic Refactoring Patterns Established

1. **Module Instantiation**: Always use keyword arguments for Pydantic BaseModel
2. **Private Attributes**: Use `PrivateAttr()` and set after instantiation, not in `__init__`
3. **Field Declarations**: Replace `__init__` assignments with field declarations
4. **Type Annotations**: Must be accurate for Pydantic validation (including Union types)
5. **Spawn Methods**: Create new instance, then set private attributes
6. **Test Mocks**: Convert mock classes to proper Pydantic models

### Design Changes from Refactoring

1. **Default to_msg Changed**: `NullToPrompt()` → `ToText()` because instruction functors return strings not Msg objects
2. **Engine Type Expanded**: Added `str` option for lazy attribute lookup pattern
3. **Error Handling**: Added validators to handle type coercion (e.g., ModuleList conversion)
4. **Property Accessors**: Added `kwargs` property to access `_kwargs` private attribute

---

## Phase 6: Outstanding Issues & Future Work 📋

### Known Warnings (Non-blocking)

1. **Underscore Parameter Names** (test_process.py::test_forward_returns_none)
   - **Warning**: `RuntimeWarning: fields may not start with an underscore, ignoring "_"`
   - **Cause**: Test uses `def forward(self, *_: Any, **__: Any)` with underscore parameter names
   - **Impact**: func_arg_model generation fails for this specific test class
   - **Current Behavior**: UserWarning issued, ProcessCall unavailable for this class, but direct method calls work
   - **TODO**: Enhance func_arg_model to handle underscore parameters gracefully
     - Option 1: Skip parameters starting with `_` (convention for unused params)
     - Option 2: Rename to `unused_args`, `unused_kwargs` in generated model
     - **Note**: Usually args starting with `_` are not required/used

2. **Field Name Shadowing** (_inst.py)
   - **Warning**: `Field name "train" in "SigF" shadows an attribute in parent "IBase"`
   - **Impact**: None - both parent and child use same field, Pydantic handles correctly
   - **Action**: Consider if `train` should be lifted to IBase or remain duplicated

### Full Integration Testing ✅ COMPLETED

**Test Suites Verified:**
- ✅ `tests/proc/` - All 353 proc module tests passing (100%)
- ⚠️ `tests_adapt/` - Adapter integration tests not run (out of scope for proc refactoring)

**Process Subclasses Verified:**
- ✅ Process, AsyncProcess, StreamProcess, AsyncStreamProcess
- ✅ Func, AsyncFunc (wrapper processes)
- ✅ Sequential, AsyncParallel (composite processes)
- ✅ Graph execution (DataFlow, T nodes, V nodes)
- ✅ Instruction decorators (signaturefunc, instructfunc, methods)
- ✅ Response processors (ToOut hierarchy)
- ✅ Critic (optimization/evaluation)

---

## Development Notes

### Architecture Decisions

**Why separate _arg_model.py?**
- Avoids import errors from DataFlow issues
- Allows isolated testing
- Cleaner module organization

**Why not wrap POSITIONAL_ONLY params?**
- They behave like regular params at runtime
- Only VAR_POSITIONAL (*args) needs special handling
- Simpler mental model

**Why KWOnly wrapper instead of kwargs dict?**
- Preserves type information
- Allows Pydantic validation
- Distinguishes keyword-only from VAR_KEYWORD

### Current Blockers
None - Phase 2 completed successfully

### Questions Resolved
- ✅ `build()` validates through Pydantic's model validation
- ✅ Partial argument application handled via default values
- ℹ️ Building from another Args instance not needed yet

---

## Session Continuity

### Current Status: ALL PHASES COMPLETE ✅

**Completed Work:**
1. ✅ Phase 1: Extract `func_arg_model` to `_arg_model.py` (36 tests)
2. ✅ Phase 2: Implement `BaseArgs.build()` class method (48 tests total)
3. ✅ Phase 3: Integrate into `_process.py` with error handling
4. ✅ Phase 4: Resolve DataFlow import errors
5. ✅ Phase 5: Fix all 43 test failures + 1 collection error
6. ✅ Phase 6: Full integration testing and documentation

**Design Changes:**
- Extracted 176 lines of arg model code to `_arg_model.py`
- Removed 240+ lines of duplicate code from `_process.py`
- Added comprehensive error handling (warnings when model generation fails)
- Fixed `_graph.py` imports, type annotations, and hashability
- Fixed `_optim.py` invalid import
- Converted all test helpers to Pydantic patterns
- Updated decorator instantiation to use keyword arguments
- Fixed all spawn() methods for Pydantic compatibility
- Changed default to_msg from NullToPrompt to ToText
- Added str support to engine field for lazy lookup

**Final Test Results:**
- ✅ 48/48 func_arg tests passing
- ✅ 353/353 proc tests passing (100% pass rate)
- ✅ No regressions detected
- ⚠️ 5 benign warnings (underscore params, field shadowing)

**Challenges Overcome:**
1. **Pydantic Strict Validation**: Required all BaseModel instantiation to use keyword args
2. **Private Attributes**: Converted underscore fields to PrivateAttr() pattern
3. **Spawn Methods**: Fixed 5 spawn() implementations to properly create Pydantic instances
4. **Type Annotations**: Updated to include all valid types (str for engine, Type[BaseModel] for struct)
5. **Test Mocks**: Converted all mock classes from __init__ pattern to field declarations
6. **Validators**: Added field validators for type coercion (ModuleList handling)

### Key Files
- **Implementation**: `dachi/proc/_arg_model.py` (arg model logic)
- **Integration**: `dachi/proc/_process.py` (Process classes with error handling)
- **Fixed**: `dachi/proc/_graph.py` (DataFlow import issues resolved)
- **Fixed**: `dachi/proc/_optim.py` (invalid import removed)
- **Tests**: `tests/proc/test_process_func_arg.py` (48 tests passing)

### Git Status
```
M  dachi/proc/__init__.py (imports updated)
M  dachi/proc/_process.py (removed 240+ duplicate lines, added error handling)
M  dachi/proc/_graph.py (fixed imports, field() usage, type annotations, hashability)
M  dachi/proc/_optim.py (removed invalid import)
M  dachi/proc/_inst.py (PrivateAttr, spawn fixes, ToText default, str engine)
M  dachi/proc/_resp.py (StructOut type fix, TupleOut validator)
M  tests/proc/test_process.py (Pydantic field declarations)
M  tests/proc/test_process_call.py (API updates)
M  tests/proc/test_graph.py (SerialDict removal, positional args, hashability)
M  tests/proc/test_resp.py (ToOut subclasses, negative tests)
M  tests/proc/test_optim.py (MockProcess Pydantic conversion)
```

**All phases complete. Refactoring successful. 353/353 tests passing.**

---

## Success Criteria

### Phase 1 ✅
- [x] func_arg_model extracted and tested
- [x] All parameter types handled correctly
- [x] Argument ordering verified
- [x] 36/36 tests passing

### Phase 2 ✅
- [x] BaseArgs.build() implemented
- [x] Comprehensive tests for build()
- [x] Edge cases handled
- [x] Documentation complete

### Phase 3 ✅
- [x] Imports from _arg_model added to _process.py
- [x] 240+ lines of duplicate code removed
- [x] Error handling added to all __init_subclass__ methods
- [x] Defensive checks added to all forward_process_call methods
- [x] BaseProcessCall.depends_on() fixed for FieldInfo
- [x] 48/48 func_arg tests passing
- [x] Integration verified

### Phase 4 ✅
- [x] DataFlow import errors resolved
- [x] _graph.py imports consolidated and fixed
- [x] _optim.py invalid import removed
- [x] dachi.proc imports successfully
- [x] 210/253 proc tests passing

### Phase 5 ✅
- [x] Fix test_process.py failures (Pydantic field declarations)
- [x] Fix test_process_call.py failures (API updates)
- [x] Fix test_instruct.py failures (11 tests - decorators, spawn, to_msg)
- [x] Fix test_optim.py failures (6 tests - MockProcess)
- [x] Fix test_resp.py failures (3 tests - TupleOut, type validation)
- [x] Fix test_graph.py collection error (SerialDict, hashability)
- [x] **Target: 353/353 proc tests passing - ACHIEVED**

### Phase 6 ✅
- [x] Full proc test suite passing
- [x] All Process subclasses verified
- [x] Graph execution tested
- [x] No regressions detected
- [x] Documentation updated

### Overall Project Status
- [x] Phases 1-6 complete
- [x] Process refactoring COMPLETE
- **Final: 353/353 tests passing (100%)**
- **Goal ACHIEVED**
