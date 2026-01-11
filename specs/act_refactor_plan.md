# Act Module Refactoring Plan

## Overview
Update `dachi/act` module after core refactoring completed. Both behavior tree (`_bt`) and state chart (`_chart`) modules must be fixed to work with the new Pydantic-based Module system.

## Status: IN PROGRESS - Phase 0 (Scope Refactoring)

---

## Investigation Summary

- **Test Files**: 18 test files in `tests/act/`
- **Module Classes**: 5 direct Module subclasses (Task, LeafTask, FTask, ChartBase, PseudoState)
- **Generic Classes**: 12 classes using generics (must be preserved exactly)
- **Test Mocks**: 4 test classes using `__init__` needing conversion
- **NEW**: Scope classes need Pydantic conversion for Runtime[Scope] compatibility

---

## Execution Plan - Detailed Todo List

### Phase 0: Convert Scope Classes to Pydantic BaseModel (PREREQUISITE)

**Why This Is Needed:**
- `BT` uses `_scope: Runtime[Scope] = PrivateRuntime(default_factory=Scope)`
- Pydantic cannot validate `Runtime[Scope]` unless `Scope` is Pydantic-compatible
- Current error: `PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'dachi.core._scope.Scope'>`
- Scope needs to be serializable in `state_dict` for proper checkpoint/restore

**Classes to Convert:**
1. `Scope` - Main hierarchical data storage class
2. `Ctx` - Context proxy for index-based access
3. `BoundScope` - Scope with external context bindings
4. `BoundCtx` - Context with variable bindings

**Key Design Decision:**
- **Parent reference must be private** to avoid circular serialization
- Use `_parent: Optional['Scope'] = pydantic.PrivateAttr(default=None)`
- Provide `@property parent` for backward compatibility
- `model_post_init` automatically reconstructs parent links after deserialization

#### 0.1 Convert Scope to Pydantic BaseModel
- **File**: `dachi/core/_scope.py:162-476`
- **Changes**:
  ```python
  class Scope(pydantic.BaseModel):
      model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

      # Public fields (serialized)
      name: Optional[str] = None
      children: Dict[str, 'Scope'] = pydantic.Field(default_factory=dict)
      full_path: Dict[Tuple, Any] = pydantic.Field(default_factory=dict)
      aliases: Dict[str, Tuple[int, ...]] = pydantic.Field(default_factory=dict)
      fields: Dict[str, Any] = pydantic.Field(default_factory=dict)

      # Private field (not serialized, runtime only)
      _parent: Optional['Scope'] = pydantic.PrivateAttr(default=None)

      def model_post_init(self, __context):
          """Reconstruct parent references for all children"""
          for child in self.children.values():
              child._parent = self

      @property
      def parent(self) -> Optional['Scope']:
          """Access parent scope"""
          return self._parent

      # All existing methods stay the same, replace self.parent with self._parent
  ```
- **Method Updates**: Replace all `self.parent` with `self._parent` in method bodies
- **Child Creation**: Update `child()` method to set `_parent` on new children

#### 0.2 Convert Ctx to Pydantic BaseModel
- **File**: `dachi/core/_scope.py:551-633`
- **Changes**:
  ```python
  class Ctx(pydantic.BaseModel):
      model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

      scope: Scope
      index_path: Tuple[int, ...]

      # All methods stay the same
  ```
- **Note**: `Ctx` holds reference to `Scope`, not the other way around - no circular ref issue

#### 0.3 Convert BoundScope to Pydantic BaseModel
- **File**: `dachi/core/_scope.py:488-549`
- **Changes**:
  ```python
  class BoundScope(Scope):
      base_scope: Scope
      base_ctx: Ctx
      bindings: Dict[str, str]

      # Override __init__ is not needed with Pydantic
      # All methods stay the same
  ```
- **Note**: Inherits from Pydantic `Scope`, gets serialization for free

#### 0.4 Convert BoundCtx to Pydantic BaseModel
- **File**: `dachi/core/_scope.py:635-705`
- **Changes**:
  ```python
  class BoundCtx(Ctx):
      bindings: Dict[str, str]
      base_ctx: Ctx

      # Override __init__ is not needed with Pydantic
      # All methods stay the same
  ```

#### 0.5 Test Scope Serialization Roundtrip
- **Create test**: `tests/core/test_scope_serialization.py`
- **Verify**:
  1. `Scope` can be created and used normally
  2. `model_dump()` serializes without circular reference errors
  3. Deserialization via `Scope(**data)` works
  4. Parent links are reconstructed correctly after deserialization
  5. `Runtime[Scope]` works in Module fields
  6. **Navigation still works**: `../`, `/target/`, lexical scoping
  7. **Bindings work**: BoundScope and BoundCtx functionality
  8. **Edge cases**: Deep hierarchies, empty scopes, no name
- **Command**: `pytest tests/core/test_scope_serialization.py -v`
- **Also run**: `pytest tests/core/test_scope.py -v` (if existing Scope tests exist)

#### 0.6 Update Scope Usage in Act Module
- **Files**: Already using correct patterns
  - `dachi/act/_bt/_roots.py:17` - `_scope: Runtime[Scope]` ✓
  - `dachi/act/_chart/_chart.py:50,68` - Direct `Scope` usage ✓
- **Action**: Verify no changes needed after Scope conversion

---

### Phase 1: Fix Blocking Import/API Issues (CRITICAL)

**These prevent the module from loading at all**

#### 1.1 Fix `anno=` parameter in Task class
- **File**: `dachi/act/_bt/_core.py:187`
- **Change**: `_status: Runtime[TaskStatus] = PrivateRuntime(default=TaskStatus.READY, anno=TaskStatus)`
- **To**: `_status: Runtime[TaskStatus] = PrivateRuntime(default=TaskStatus.READY)`
- **Reason**: `anno` parameter was removed from `PrivateRuntime` during proc refactoring

#### 1.2 Fix `anno=` parameter in FTask class
- **File**: `dachi/act/_bt/_core.py:540`
- **Change**: `_obj: Runtime[t.Any] = PrivateRuntime(anno=t.Any)`
- **To**: `_obj: Runtime[t.Any] = PrivateRuntime(default=None)`
- **Reason**: Same as above

#### 1.3 Remove broken imports from decorators
- **File**: `dachi/act/_bt/_decorators.py:6-7`
- **Remove lines**:
  ```python
  from ._core import Task, TaskStatus, CompositeTask, LeafTask, RestrictedTaskSchemaMixin, TASK, LEAF
  from dachi.core._base import filter_class_variants
  ```
- **Replace with**:
  ```python
  from ._core import Task, TaskStatus, CompositeTask, LeafTask, TASK, LEAF
  ```
- **Reason**: `RestrictedTaskSchemaMixin` and `filter_class_variants` don't exist (only used in commented code)

#### 1.4 Remove broken import from chart
- **File**: `dachi/act/_chart/_chart.py:9`
- **Change**: `from dachi.core import Runtime, ModuleList, PrivateParam, PrivateRuntime, ParamField`
- **To**: `from dachi.core import Runtime, ModuleList, PrivateParam, PrivateRuntime`
- **Reason**: `ParamField` doesn't exist

#### 1.5 Comment out incomplete FTask class
- **File**: `dachi/act/_bt/_core.py:533-565`
- **Action**: Comment out entire `FTask` class definition
- **Reason**: Incomplete implementation that references missing methods

#### 1.6 Comment out FTask test helper
- **File**: `tests/act/test_core.py:111-122`
- **Action**: Comment out `_CountingFTask` class
- **Reason**: Depends on FTask which we're commenting out

#### 1.7 Verify imports work
- **Command**: `python -c "from dachi.act import Task"`
- **Expected**: No errors
- **If errors**: Fix before proceeding to Phase 2

---

### Phase 2: Fix Test Mock Classes Using `__init__`

**Convert 4 test helper classes to Pydantic-style field declarations**

#### 2.1 Fix `_DummyTask` in test_core.py
- **File**: `tests/act/test_core.py:94-108`
- **Current**:
  ```python
  class _DummyTask(core.Task):
      def __init__(self, ret_status: core.TaskStatus):
          super().__init__()
          self._ret_status = ret_status
  ```
- **Convert to**:
  ```python
  class _DummyTask(core.Task):
      ret_status: core.TaskStatus
      _ret_status: core.TaskStatus = pydantic.PrivateAttr()

      def model_post_init(self, __context):
          super().model_post_init(__context)
          self._ret_status = self.ret_status
  ```
- **Reason**: Pydantic models require keyword arguments, can't use positional `__init__`

#### 2.2 Fix `ToggleWait` in test_leafs.py
- **File**: `tests/act/test_leafs.py:193-204`
- **Current**:
  ```python
  class ToggleWait(WaitCondition):
      def __init__(self):
          super().__init__()
          self._first = True
  ```
- **Convert to**:
  ```python
  class ToggleWait(WaitCondition):
      _first: bool = pydantic.PrivateAttr(default=True)
  ```
- **Reason**: Private state should use `PrivateAttr`, no need for `__init__`

#### 2.3 Fix `_SyncBoolProc` in utils.py
- **File**: `tests/act/utils.py:109-115`
- **Current**:
  ```python
  class _SyncBoolProc(Process):
      def __init__(self, val: Any) -> None:
          self._val = val

      def forward(self) -> Any:
          return self._val
  ```
- **Convert to**:
  ```python
  class _SyncBoolProc(Process):
      val: Any

      def forward(self) -> Any:
          return self.val
  ```
- **Reason**: Process already uses Pydantic; use field instead of private `__init__`

#### 2.4 Fix `SimpleOutputAction` in test_serial.py
- **File**: `tests/act/test_serial.py:436-445`
- **Current**:
  ```python
  class SimpleOutputAction(Action):
      class outputs:
          value: int

      def __init__(self, output_value: int = 42):
          super().__init__()
          self._output_value = output_value
  ```
- **Convert to**:
  ```python
  class SimpleOutputAction(Action):
      output_value: int = 42

      class outputs:
          value: int
  ```
- **And update execute method**: Use `self.output_value` instead of `self._output_value`
- **Reason**: Make it a proper Pydantic field

---

### Phase 3: Fix `model_post_init` Signatures

**Ensure all `model_post_init` methods include `__context` parameter**

#### 3.1 Fix FTask.model_post_init signature
- **File**: `dachi/act/_bt/_core.py:542`
- **Change**: `def model_post_init(self):`
- **To**: `def model_post_init(self, __context):`
- **Note**: This is inside the FTask class we're commenting out, but fix for completeness
- **Reason**: Pydantic requires `__context` parameter

---

### Phase 4: Test & Validate

**Run tests incrementally to catch and fix issues early**

#### 4.1 Test basic imports
- **Command**: `python -c "from dachi.act._bt._core import Task, TaskStatus, LeafTask"`
- **Expected**: No errors
- **If fails**: Review Phase 1 changes

#### 4.2 Test state chart imports
- **Command**: `python -c "from dachi.act._chart import StateChart"`
- **Expected**: No errors
- **If fails**: Check Phase 1.4 fix

#### 4.3 Run core tests
- **Command**: `pytest tests/act/test_core.py -v`
- **Expected**: All tests pass
- **If fails**: Check Phase 2.1 conversion

#### 4.4 Run leafs tests
- **Command**: `pytest tests/act/test_leafs.py -v`
- **Expected**: All tests pass
- **If fails**: Check Phase 2.2 conversion

#### 4.5 Run serial tests
- **Command**: `pytest tests/act/test_serial.py -v`
- **Expected**: All tests pass
- **If fails**: Check Phase 2.4 conversion

#### 4.6 Run full act test suite
- **Command**: `pytest tests/act -v`
- **Expected**: All 18 test files pass, 0 errors
- **Count tests**: Should be consistent with baseline

#### 4.7 Fix any remaining issues
- **Common issues from proc refactoring**:
  - Type annotation validation errors → Add None to Union types
  - Missing keyword arguments → Convert positional to keyword
  - Field access on ShareableItem → Use `.data` accessor
- **Strategy**: Fix one test file at a time, run tests after each fix

---

## Special Considerations

### DO NOT MODIFY: Generic Classes (12 total)
These must be preserved exactly as-is:
- `CompositeTask(Task, t.Generic[TASK])`
- `Serial(CompositeTask[TASK], t.Generic[TASK])`
- `Decorator(CompositeTask, t.Generic[TASK])`
- `BoundTask(Task, t.Generic[LEAF])`
- `BT(Task, t.Generic[TASK])`
- `MultiTask(ParallelTask[TASK], t.Generic[TASK])`
- `PreemptCond(Serial[TASK], t.Generic[TASK, CONDITION])`
- `StateChart(ChartBase, ChartEventHandler, t.Generic[V])`
- `CompositeState(BaseState, ChartEventHandler, Recoverable, t.Generic[V])`
- `Region(ChartBase, ChartEventHandler, Recoverable, t.Generic[V])`
- `BoundState(BaseState, t.Generic[STATE])`
- `BoundStreamState(BaseState, t.Generic[STREAM_STATE])`

**Why**: Generics are used for Pydantic discrimination and type safety. Breaking them will break serialization.

### DO NOT REMOVE: Commented Code (for now)
Large blocks of commented code should stay:
- Old `__post_init__` implementations
- `restricted_schema` methods
- State machine code marked with TODO

**Discuss before removing**: If cleanup is desired later

### ADD Import Where Needed
- `import pydantic` in test files that use `PrivateAttr`

---

## Success Criteria

✅ All imports work without TypeError
✅ All 18 test files run without collection errors
✅ All tests pass (target: 100% pass rate)
✅ Zero breaking API changes to public interfaces
✅ All generic types preserved exactly
✅ Consistent patterns with proc module refactoring

---

## Estimated Impact

### Files to Modify
- **Core infrastructure**: 1 (`dachi/core/_scope.py` - Scope/Ctx/BoundScope/BoundCtx conversion)
- **Core module files**: 3 (`_core.py`, `_decorators.py`, `_chart.py`)
- **Test files**: 4 (`test_core.py`, `test_leafs.py`, `utils.py`, `test_serial.py`)
- **New test file**: 1 (`tests/core/test_scope_serialization.py`)
- **Total**: 9 files

### Lines Changed
- **Phase 0 (Scope)**: ~100 lines (convert 4 classes to Pydantic, replace `self.parent` with `self._parent`)
- **Phase 1-3**: ~35 lines (anno params, broken imports, test conversions, comments)
- **Tests**: ~50 lines (new serialization tests)
- **Total**: ~185 lines modified/added

---

## Dependencies Between Phases

- **Phase 0 is BLOCKING** - Must complete before Phase 1.7 (import verification)
- **Phases 1.1-1.6** can proceed independently (simple code fixes)
- **Phase 0 completion required** before continuing to Phase 2+

**Recommended approach**: Complete Phase 0 fully, then resume with Phase 1.7+

---

## Risk Assessment

### Phase 0 Risks (Scope Conversion)
- **High Impact**: Scope is core infrastructure used throughout act module
- **Mitigation**: Extensive testing, incremental changes, verify existing tests pass
- **Fallback**: Revert Scope changes if issues found, use alternative approach

### Phase 1-4 Risks (Act Module Fixes)
- **Medium Impact**: Mostly isolated to act module
- **Mitigation**: Test after each phase, fix incrementally
- **Fallback**: Phase-specific rollback possible

---

## Rollback Plan

If critical issues arise:
1. **Git checkout** affected files
2. **Document specific failure** in this plan
3. **Consult with framework design** before retrying
4. **Consider partial rollback**:
   - Phase 0 only: Revert `_scope.py`, explore alternative Scope handling
   - Phase 1-4 only: Revert act module changes, keep Phase 0
   - Full rollback: Revert all changes if fundamental issue discovered

---

## Notes from Proc Module Refactoring

### Patterns That Worked
1. **Keyword arguments everywhere**: Pydantic requires them
2. **PrivateAttr for internal state**: Not part of spec/serialization
3. **Field declarations**: Instead of `__init__` assignments
4. **model_post_init with __context**: Required signature
5. **Two-step initialization**: Create instance, then set private attrs in model_post_init

### Common Errors to Watch For
- `TypeError: got unexpected keyword argument 'anno'`
- `ImportError: cannot import name 'ParamField'`
- `TypeError: BaseModel.__init__() takes 1 positional argument`
- `ValidationError` on Union types needing `None`
- Field access on ShareableItem without `.data`

---

**Ready to execute**: All blocking issues identified, conversion patterns clear, test strategy defined.
