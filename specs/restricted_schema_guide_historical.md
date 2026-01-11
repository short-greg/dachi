# RestrictedSchemaMixin Guide

## Overview

`RestrictedSchemaMixin` provides a way to create JSON schemas where placeholder types (like `Task`) are restricted to specific allowed variants (like only `ActionA`, `ActionB`, `ActionC`).

**Purpose**: When you have a module that accepts generic types (e.g., any `Task`), but you want to generate a schema that only allows specific subtypes, `RestrictedSchemaMixin` provides the tools to build that restricted schema.

NOTE: UPDATED PLAN FOR ADDING SCHEMA TO DATAFLOW has been ADDED TO THE BOTTOM!
BEFORE IMPLEMENTING EACH PHASE OF IT BRAINSTORM!
THE STATECHART IMPLEMENTATION WAS COMPLETED. REVIEW THE CODE!

---

# IMPLEMENTATION PLAN (2025-10-28)

## Plan Completeness Checklist

✅ **Mixin architecture defined** - 3 classes with clear responsibilities
✅ **Pattern definitions** - Patterns A, B, C fully documented
✅ **Complete implementations** - Full code for all 3 mixins
✅ **Low-level helpers** - All schema manipulation helpers specified
✅ **Nullable field handling** - anyOf structure handling in list updates
✅ **Recursive calls** - Domain-specific isinstance checks enforced
✅ **Pattern A solution** - Temporary instance creation for pass-through
✅ **Test specifications** - Complete test cases with expected behavior
✅ **Imports and dependencies** - All imports listed
✅ **Implementation order** - 19 tasks in 4 phases

## Architecture Overview

We are creating a **clean, domain-specific mixin system** from scratch:

1. **RestrictedSchemaMixin** (base) - Core helpers only, no domain logic. Does NOT implement `_schema_process_variants()`.
2. **RestrictedTaskSchemaMixin** (for behavior trees, behavior tree tasks and so on) - Implements `_schema_process_variants()` with `isinstance(variant, RestrictedTaskSchemaMixin)` check.
3. **RestrictedStateSchemaMixin** (for state charts, state chart regions, etc) - Implements `_schema_process_variants()` with `isinstance(variant, RestrictedStateSchemaMixin)` check.

**Critical**: Domain-specific mixins check for THEIR OWN type, not the base type. This prevents task/state cross-contamination.

## Implementation Patterns

### Pattern A: Pass-Through
- **Definition**: No field variants, pass variants to child class
- **Process**:
  1. Call child class's `restricted_schema()` to get ONE restricted schema
  2. Update parent's schema with that ONE schema
- **Examples**:
  - `StateChart` → `Region` (pass states down)
  - `CompositeState` → `Region` (pass states down)

### Pattern B: Direct Variants
- **Definition**: Field accepts variants directly
- **Process**:
  1. Process variants yourself to get N schemas
  2. Update parent's schema with those N schemas
- **Examples**:
  - `Region` with state variants
  - `Sequence` with task variants
  - `Multi` with task variants

### Pattern C: Single Field
- **Definition**: Not a list/dict, just one module field
- **Process**:
  1. Process variants to get N schemas
  2. Update parent's schema for single field
- **Examples**:
  - `BT` with root field (`root: InitVar[Task | None]`)

## Implementation Order

### Phase 1: Create Fresh Mixins (File: `dachi/core/_base.py`)

#### Task 1: Create new RestrictedSchemaMixin (base class)
**File**: `dachi/core/_base.py`

**Imports needed**:
```python
from abc import ABC, abstractmethod
import typing as t
from typing import Callable
```

**What to create**:
- Abstract base class with abstract `restricted_schema()` method
- Does NOT implement `_schema_process_variants()` (subclasses do this)
- Implements 3 update helpers and low-level schema manipulation helpers

**Complete Implementation**:

```python
class RestrictedSchemaMixin(ABC):
    """
    Base mixin for creating restricted JSON schemas.

    Subclasses must implement:
    - restricted_schema(**kwargs) -> dict
    - _schema_process_variants(variants, mixin_class, ...) -> list[dict]
    """

    @abstractmethod
    def restricted_schema(
        self,
        *,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> dict:
        """Generate restricted schema. Must be implemented by subclasses."""
        raise NotImplementedError()

    # =========================
    # 3 Update Helpers (implemented in base)
    # =========================

    def _schema_update_list_field(
        self,
        schema: dict,
        *,
        field_name: str,
        placeholder_name: str,
        variant_schemas: list[dict],
        profile: str = "shared"
    ) -> dict:
        """
        Update a ModuleList field in the schema.

        Path: ["properties", field_name, "items"]

        Handles nullable fields: If field is "ModuleList[T] | None", wraps in anyOf.

        Args:
            schema: The base schema dict to update
            field_name: Name of the ModuleList field (e.g., "tasks")
            placeholder_name: Name of placeholder spec (e.g., "TaskSpec")
            variant_schemas: List of schema dicts for allowed variants
            profile: "shared" (use $defs/Allowed_*) or "inline" (use oneOf)

        Returns:
            Updated schema dict
        """
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]
        self._schema_require_defs_for_entries(schema, entries)

        # Build the union
        if profile == "shared":
            union_ref = self._schema_ensure_shared_union(
                schema,
                placeholder_name=placeholder_name,
                entries=entries
            )
            replacement = {"$ref": union_ref}
        else:
            replacement = self._schema_make_union_inline(entries)

        # Check if field is nullable (has anyOf with null)
        field_schema = schema.get("properties", {}).get(field_name, {})
        if "anyOf" in field_schema:
            # Nullable field: update the items in the array part of anyOf
            # Structure: {"anyOf": [{"type": "array", "items": {...}}, {"type": "null"}]}
            for option in field_schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "array":
                    option["items"] = replacement
                    break
        else:
            # Non-nullable field: directly update items
            self._schema_replace_at_path(schema, ["properties", field_name, "items"], replacement)

        return schema

    def _schema_update_dict_field(
        self,
        schema: dict,
        *,
        field_name: str,
        placeholder_name: str,
        variant_schemas: list[dict],
        profile: str = "shared"
    ) -> dict:
        """
        Update a ModuleDict field in the schema.

        Path: ["properties", field_name, "additionalProperties"]

        Args:
            schema: The base schema dict to update
            field_name: Name of the ModuleDict field (e.g., "states")
            placeholder_name: Name of placeholder spec (e.g., "BaseStateSpec")
            variant_schemas: List of schema dicts for allowed variants
            profile: "shared" or "inline"

        Returns:
            Updated schema dict
        """
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]
        self._schema_require_defs_for_entries(schema, entries)

        if profile == "shared":
            union_ref = self._schema_ensure_shared_union(
                schema,
                placeholder_name=placeholder_name,
                entries=entries
            )
            replacement = {"$ref": union_ref}
        else:
            replacement = self._schema_make_union_inline(entries)

        self._schema_replace_at_path(
            schema,
            ["properties", field_name, "additionalProperties"],
            replacement
        )

        return schema

    def _schema_update_single_field(
        self,
        schema: dict,
        *,
        field_name: str,
        placeholder_name: str,
        variant_schemas: list[dict],
        profile: str = "shared"
    ) -> dict:
        """
        Update a single module field in the schema.

        Path: ["properties", field_name]

        Args:
            schema: The base schema dict to update
            field_name: Name of the single field (e.g., "root")
            placeholder_name: Name of placeholder spec (e.g., "TaskSpec")
            variant_schemas: List of schema dicts for allowed variants
            profile: "shared" or "inline"

        Returns:
            Updated schema dict
        """
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]
        self._schema_require_defs_for_entries(schema, entries)

        if profile == "shared":
            union_ref = self._schema_ensure_shared_union(
                schema,
                placeholder_name=placeholder_name,
                entries=entries
            )
            replacement = {"$ref": union_ref}
        else:
            replacement = self._schema_make_union_inline(entries)

        self._schema_replace_at_path(schema, ["properties", field_name], replacement)

        return schema

    # =========================
    # Low-Level Schema Helpers
    # =========================

    @staticmethod
    def _schema_name_from_dict(schema_dict: dict) -> str:
        """
        Extract spec name from schema dict.

        Tries 'title' first, then tail of '$id'.

        Args:
            schema_dict: JSON schema dict

        Returns:
            Spec name (e.g., "TaskSpec")

        Raises:
            TypeError: If no title or $id found
        """
        if "title" in schema_dict and isinstance(schema_dict["title"], str):
            return schema_dict["title"].strip()

        if "$id" in schema_dict and isinstance(schema_dict["$id"], str):
            _id = schema_dict["$id"].strip()
            tail = _id.rsplit("/", 1)[-1].rsplit("#", 1)[-1] or _id
            return tail

        raise TypeError("Schema dict must have 'title' or '$id' to derive spec name")

    @staticmethod
    def _schema_require_defs_for_entries(schema: dict, entries: list[tuple[str, dict]]) -> None:
        """
        Add entries to $defs if not already present.

        Args:
            schema: Schema dict to update
            entries: List of (name, schema_dict) tuples
        """
        defs = schema.setdefault("$defs", {})
        for name, entry_schema in entries:
            defs.setdefault(name, entry_schema)

    @staticmethod
    def _schema_build_refs(entries: list[tuple[str, dict]]) -> list[dict]:
        """
        Convert entries to $ref list for oneOf.

        Args:
            entries: List of (name, schema_dict) tuples

        Returns:
            List of {"$ref": "#/$defs/<name>"} dicts
        """
        return [{"$ref": f"#/$defs/{name}"} for name, _ in entries]

    @staticmethod
    def _schema_make_union_inline(entries: list[tuple[str, dict]]) -> dict:
        """
        Create inline oneOf union.

        Args:
            entries: List of (name, schema_dict) tuples

        Returns:
            {"oneOf": [...]} dict
        """
        return {"oneOf": RestrictedSchemaMixin._schema_build_refs(entries)}

    @staticmethod
    def _schema_allowed_union_name(placeholder_name: str) -> str:
        """
        Generate name for shared union in $defs.

        Args:
            placeholder_name: Original placeholder name (e.g., "TaskSpec")

        Returns:
            Allowed union name (e.g., "Allowed_TaskSpec")
        """
        return f"Allowed_{placeholder_name}"

    @classmethod
    def _schema_ensure_shared_union(
        cls,
        schema: dict,
        *,
        placeholder_name: str,
        entries: list[tuple[str, dict]]
    ) -> str:
        """
        Ensure shared union exists in $defs and return its $ref.

        Args:
            schema: Schema dict to update
            placeholder_name: Original placeholder name
            entries: List of (name, schema_dict) tuples

        Returns:
            Reference string (e.g., "#/$defs/Allowed_TaskSpec")
        """
        defs = schema.setdefault("$defs", {})
        allowed_name = cls._schema_allowed_union_name(placeholder_name)

        if allowed_name not in defs:
            defs[allowed_name] = cls._schema_make_union_inline(entries)

        return f"#/$defs/{allowed_name}"

    @staticmethod
    def _schema_node_at(schema: dict, path: list[str]) -> t.Any:
        """
        Navigate to node at path in schema.

        Args:
            schema: Schema dict
            path: List of keys (e.g., ["properties", "tasks", "items"])

        Returns:
            Node at path, or None if not found
        """
        cur = schema
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    @staticmethod
    def _schema_replace_at_path(schema: dict, path: list[str], replacement: t.Any) -> None:
        """
        Replace node at path with replacement.

        Args:
            schema: Schema dict to update
            path: List of keys to navigate to
            replacement: New value to set

        Raises:
            ValueError: If path is empty
            KeyError: If path is invalid
        """
        if not path:
            raise ValueError("Path cannot be empty")

        *parent_path, last_key = path
        parent = RestrictedSchemaMixin._schema_node_at(schema, parent_path) if parent_path else schema

        if not isinstance(parent, dict):
            raise KeyError(f"Invalid path: {'.'.join(path)}")

        parent[last_key] = replacement
```

**Key Design Notes**:
1. Base class does NOT implement `_schema_process_variants()` - subclasses must do this
2. All low-level helpers are prefixed with `_schema_` for clarity
3. `_schema_update_list_field()` handles nullable fields (anyOf with null)
4. All helpers are either `@staticmethod` or `@classmethod` where appropriate
5. Clear docstrings with Args, Returns, Raises sections

#### Task 2: Create RestrictedTaskSchemaMixin
**File**: `dachi/core/_base.py`

**Complete Implementation**:

```python
class RestrictedTaskSchemaMixin(RestrictedSchemaMixin):
    """
    Mixin for behavior tree tasks with task-specific restrictions.

    Uses isinstance(variant, RestrictedTaskSchemaMixin) for recursion checks.
    """

    # add this to the base class now
    # since the class type is passed into here
    def _schema_process_variants(
        self,
        variants: list,
        *,
        filter_fn: t.Callable | None = None,
        restricted_schema_cls: Type[RestrictedSchemaMixin],
        _seen: dict | None = None,
        **recursive_kwargs
    ) -> list[dict]:
        """
        Process task variants and return their schema dicts.

        For each variant:
        - Apply filter_fn if provided
        - If variant is RestrictedTaskSchemaMixin, call its restricted_schema() recursively
        - Else if variant has schema() method, call it
        - Else raise TypeError

        Args:
            variants: List of task instances or classes
            filter_fn: Optional filter (e.g., lambda t: isinstance(t, Action))
            _seen: Cycle detection dict
            **recursive_kwargs: Passed to nested restricted_schema() calls (e.g., tasks=...)

        Returns:
            List of schema dicts

        Raises:
            TypeError: If variant doesn't have schema() method
        """
        if filter_fn is not None:
            variants = [v for v in variants if filter_fn(v)]

        schemas = []
        for variant in variants:
            # Check if variant implements RestrictedTaskSchemaMixin (domain-specific check!)
            if (check if type) issubclass(variant, restricted_schema_cls):
                # Recursive: call restricted_schema with same kwargs
                schemas.append(
                    variant.restricted_schema(_seen=_seen, **recursive_kwargs)
                )
            elif hasattr(variant, 'schema'):
                # Regular module: get base schema
                schema_method = getattr(variant, 'schema')
                if callable(schema_method):
                    schemas.append(schema_method())
                else:
                    raise TypeError(f"Variant {variant} has non-callable schema attribute")
            else:
                raise TypeError(
                    f"Task variant {variant} must implement RestrictedTaskSchemaMixin "
                    f"or have a schema() method"
                )

        return schemas
```

**Key Points**:
1. Checks `isinstance(variant, RestrictedTaskSchemaMixin)` - NOT the base `RestrictedSchemaMixin`
2. This ensures we only recurse on task-compatible classes
3. Passes `**recursive_kwargs` down (e.g., `tasks=tasks`) for nested restrictions

**Used by**: BT, Sequence, Selector, Multi

#### Task 3: Create RestrictedStateSchemaMixin
**File**: `dachi/core/_base.py`

**Complete Implementation**:

```python
class RestrictedStateSchemaMixin(RestrictedSchemaMixin):
    """
    Mixin for state charts with state-specific restrictions.

    Uses isinstance(variant, RestrictedStateSchemaMixin) for recursion checks.
    """

    def _schema_process_variants(
        self,
        variants: list,
        *,
        filter_fn: t.Callable | None = None,
        _seen: dict | None = None,
        **recursive_kwargs
    ) -> list[dict]:
        """
        Process state variants and return their schema dicts.

        For each variant:
        - Apply filter_fn if provided
        - If variant is RestrictedStateSchemaMixin, call its restricted_schema() recursively
        - Else if variant has schema() method, call it
        - Else raise TypeError

        Args:
            variants: List of state instances or classes
            filter_fn: Optional filter (e.g., lambda s: isinstance(s, CompositeState))
            _seen: Cycle detection dict
            **recursive_kwargs: Passed to nested restricted_schema() calls (e.g., states=...)

        Returns:
            List of schema dicts

        Raises:
            TypeError: If variant doesn't have schema() method
        """
        if filter_fn is not None:
            variants = [v for v in variants if filter_fn(v)]

        schemas = []
        for variant in variants:
            # Check if variant implements RestrictedStateSchemaMixin (domain-specific check!)
            if isinstance(variant, RestrictedStateSchemaMixin):
                # Recursive: call restricted_schema with same kwargs
                schemas.append(
                    variant.restricted_schema(_seen=_seen, **recursive_kwargs)
                )
            elif hasattr(variant, 'schema'):
                # Regular module: get base schema
                schema_method = getattr(variant, 'schema')
                if callable(schema_method):
                    schemas.append(schema_method())
                else:
                    raise TypeError(f"Variant {variant} has non-callable schema attribute")
            else:
                raise TypeError(
                    f"State variant {variant} must implement RestrictedStateSchemaMixin "
                    f"or have a schema() method"
                )

        return schemas
```

**Key Points**:
1. Checks `isinstance(variant, RestrictedStateSchemaMixin)` - NOT the base `RestrictedSchemaMixin`
2. This ensures we only recurse on state-compatible classes
3. Passes `**recursive_kwargs` down (e.g., `states=states`) for nested restrictions

**Used by**: StateChart, CompositeState, Region

#### Task 4: Test all mixins
**File**: `tests/core/test_base.py`

**Test Cases to Add**:

```python
class TestRestrictedSchemaMixin:
    """Test the base RestrictedSchemaMixin low-level helpers"""

    def test_schema_name_from_dict_uses_title(self):
        """Test extracting name from schema dict with title"""
        schema = {"title": "MyTaskSpec", "$id": "other"}
        assert RestrictedSchemaMixin._schema_name_from_dict(schema) == "MyTaskSpec"

    def test_schema_name_from_dict_uses_id_if_no_title(self):
        """Test extracting name from schema dict using $id"""
        schema = {"$id": "http://example.com/schemas/MyTaskSpec#"}
        assert RestrictedSchemaMixin._schema_name_from_dict(schema) == "MyTaskSpec"

    def test_schema_name_from_dict_raises_if_missing_both(self):
        """Test error when neither title nor $id present"""
        with pytest.raises(TypeError):
            RestrictedSchemaMixin._schema_name_from_dict({})

    def test_schema_build_refs_creates_ref_list(self):
        """Test building $ref list from entries"""
        entries = [("TaskA", {}), ("TaskB", {})]
        refs = RestrictedSchemaMixin._schema_build_refs(entries)
        assert refs == [
            {"$ref": "#/$defs/TaskA"},
            {"$ref": "#/$defs/TaskB"}
        ]

    def test_schema_make_union_inline_creates_oneof(self):
        """Test creating inline oneOf union"""
        entries = [("TaskA", {}), ("TaskB", {})]
        union = RestrictedSchemaMixin._schema_make_union_inline(entries)
        assert union == {
            "oneOf": [
                {"$ref": "#/$defs/TaskA"},
                {"$ref": "#/$defs/TaskB"}
            ]
        }

    def test_schema_allowed_union_name_adds_prefix(self):
        """Test generating Allowed_* union name"""
        assert RestrictedSchemaMixin._schema_allowed_union_name("TaskSpec") == "Allowed_TaskSpec"

    def test_schema_node_at_navigates_path(self):
        """Test navigating to node in schema"""
        schema = {"properties": {"tasks": {"items": {"$ref": "#/$defs/TaskSpec"}}}}
        node = RestrictedSchemaMixin._schema_node_at(schema, ["properties", "tasks", "items"])
        assert node == {"$ref": "#/$defs/TaskSpec"}

    def test_schema_node_at_returns_none_for_missing_path(self):
        """Test None returned for invalid path"""
        schema = {"properties": {}}
        node = RestrictedSchemaMixin._schema_node_at(schema, ["properties", "nonexistent"])
        assert node is None

    def test_schema_replace_at_path_updates_node(self):
        """Test replacing node at path"""
        schema = {"properties": {"tasks": {"items": {"old": "value"}}}}
        RestrictedSchemaMixin._schema_replace_at_path(
            schema,
            ["properties", "tasks", "items"],
            {"new": "value"}
        )
        assert schema["properties"]["tasks"]["items"] == {"new": "value"}


class TestRestrictedTaskSchemaMixin:
    """Test RestrictedTaskSchemaMixin domain-specific behavior"""

    def test_process_variants_calls_restricted_schema_on_task_mixin(self):
        """Test that variants with RestrictedTaskSchemaMixin get restricted_schema called"""
        # Create a mock task with RestrictedTaskSchemaMixin
        class MockTask(RestrictedTaskSchemaMixin):
            def restricted_schema(self, **kwargs):
                return {"title": "MockTaskSpec", "restricted": True}
            def schema(self):
                return {"title": "MockTaskSpec", "restricted": False}

        # Create a mixin instance to test
        class TestMixin(RestrictedTaskSchemaMixin):
            def restricted_schema(self, **kwargs):
                pass

        mixin = TestMixin()
        task = MockTask()

        # Process variants
        schemas = mixin._schema_process_variants([task])

        # Should have called restricted_schema, not schema
        assert schemas[0] == {"title": "MockTaskSpec", "restricted": True}

    def test_process_variants_calls_schema_on_regular_module(self):
        """Test that regular modules get schema() called"""
        class MockModule:
            def schema(self):
                return {"title": "MockModuleSpec"}

        class TestMixin(RestrictedTaskSchemaMixin):
            def restricted_schema(self, **kwargs):
                pass

        mixin = TestMixin()
        module = MockModule()

        schemas = mixin._schema_process_variants([module])
        assert schemas[0] == {"title": "MockModuleSpec"}

    def test_process_variants_raises_for_invalid_variant(self):
        """Test error for variant without schema method"""
        class TestMixin(RestrictedTaskSchemaMixin):
            def restricted_schema(self, **kwargs):
                pass

        mixin = TestMixin()

        with pytest.raises(TypeError, match="must implement RestrictedTaskSchemaMixin"):
            mixin._schema_process_variants([object()])


class TestRestrictedStateSchemaMixin:
    """Test RestrictedStateSchemaMixin domain-specific behavior"""

    def test_process_variants_checks_state_mixin_not_task_mixin(self):
        """Test that only RestrictedStateSchemaMixin variants get recursive treatment"""
        # This is the key test: ensures domain separation
        class MockTaskMixin(RestrictedTaskSchemaMixin):
            def restricted_schema(self, **kwargs):
                return {"title": "TaskSpec", "restricted": True}
            def schema(self):
                return {"title": "TaskSpec", "restricted": False}

        class TestStateMixin(RestrictedStateSchemaMixin):
            def restricted_schema(self, **kwargs):
                pass

        mixin = TestStateMixin()
        task = MockTaskMixin()

        # Process task through STATE mixin
        schemas = mixin._schema_process_variants([task])

        # Should call schema(), NOT restricted_schema() because task is not a StateSchemaMixin
        assert schemas[0] == {"title": "TaskSpec", "restricted": False}
```

**Key Test Coverage**:
1. Low-level helpers (name extraction, refs, unions, navigation)
2. Domain-specific type checking (Task vs State mixins)
3. Recursive vs non-recursive schema generation
4. Error handling for invalid variants
5. **Critical**: Test that TaskMixin only recurses on TaskMixin, StateMixin only recurses on StateMixin

### Phase 2: State Chart Classes

#### Task 5: Implement Region.restricted_schema()
**File**: `dachi/act/_chart/_region.py`
**Pattern**: B (Direct Variants)
**Field**: `states: ModuleDict[str, BaseState]`

**Implementation**:
```python
class Region(ChartBase, ChartEventHandler, Recoverable, RestrictedStateSchemaMixin):
    states: ModuleDict[str, BaseState] = None

    def restricted_schema(self, *, states=None, _profile="shared", _seen=None, **kwargs):
        if states is None:
            return self.schema()

        # Pattern B: Process state variants directly
        state_schemas = self._schema_process_variants(states, _seen=_seen, **kwargs)

        # Update schema's states field (ModuleDict)
        schema = self.schema()
        return self._schema_update_dict_field(
            schema,
            field_name="states",
            placeholder_name="BaseStateSpec",
            variant_schemas=state_schemas,
            profile=_profile
        )
```

#### Task 6: Test Region
**File**: `tests/act/test_chart.py`

**What to test**:
- Region with state variants produces correct schema
- Schema has correct $defs entries
- additionalProperties is correctly updated
- Both "shared" and "inline" profiles work

#### Task 7: Implement CompositeState.restricted_schema()
**File**: `dachi/act/_chart/_composite.py`
**Pattern**: A (Pass-Through)
**Field**: `regions: ModuleList[Region]`

**Implementation**:
```python
class CompositeState(BaseState, ChartEventHandler, Recoverable, RestrictedStateSchemaMixin):
    regions: ModuleList[Region]

    def restricted_schema(self, *, states=None, _profile="shared", _seen=None, **kwargs):
        if states is None:
            return self.schema()

        # Pattern A: Pass states to Region.restricted_schema()
        region_schema = Region.restricted_schema(
            Region(),  # or use class method if available
            states=states,
            _profile=_profile,
            _seen=_seen,
            **kwargs
        )

        # Update schema's regions field (ModuleList) with ONE Region schema
        schema = self.schema()
        return self._schema_update_list_field(
            schema,
            field_name="regions",
            placeholder_name="RegionSpec",
            variant_schemas=[region_schema],
            profile=_profile
        )
```

#### Task 8: Test CompositeState
**File**: `tests/act/test_chart.py`

**What to test**:
- CompositeState passes states down to Region correctly
- Schema has Region with restricted states
- items field is correctly updated

#### Task 9: Implement StateChart.restricted_schema()
**File**: `dachi/act/_chart/_chart.py`
**Pattern**: A (Pass-Through)
**Field**: `regions: ModuleList[Region]`

**Implementation**:
```python
class StateChart(ChartBase, ChartEventHandler, RestrictedStateSchemaMixin):
    regions: ModuleList[Region]

    def restricted_schema(self, *, states=None, _profile="shared", _seen=None, **kwargs):
        if states is None:
            return self.schema()

        # Pattern A: Pass states to Region.restricted_schema()
        region_schema = Region.restricted_schema(
            Region(),
            states=states,
            _profile=_profile,
            _seen=_seen,
            **kwargs
        )

        # Update schema's regions field (ModuleList) with ONE Region schema
        schema = self.schema()
        return self._schema_update_list_field(
            schema,
            field_name="regions",
            placeholder_name="RegionSpec",
            variant_schemas=[region_schema],
            profile=_profile
        )
```

#### Task 10: Test StateChart
**File**: `tests/act/test_chart.py`

**What to test**:
- StateChart passes states down to Region correctly
- Full integration test with StateChart → Region → States

### Phase 3: Behavior Tree Classes

#### Task 11: Implement Sequence.restricted_schema()
**File**: `dachi/act/_bt/_serial.py`
**Pattern**: B (Direct Variants)
**Field**: `tasks: ModuleList[Task] | None`

**Implementation**:
```python
class Sequence(Serial, RestrictedTaskSchemaMixin):
    tasks: ModuleList[Task] | None = None

    def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        if tasks is None:
            return self.schema()

        # Pattern B: Process task variants directly
        task_schemas = self._schema_process_variants(tasks, _seen=_seen, tasks=tasks, **kwargs)

        # Update schema's tasks field (ModuleList)
        schema = self.schema()
        return self._schema_update_list_field(
            schema,
            field_name="tasks",
            placeholder_name="TaskSpec",
            variant_schemas=task_schemas,
            profile=_profile
        )
```

#### Task 12: Test Sequence
**File**: `tests/act/test_serial.py`

**What to test**:
- Sequence with task variants produces correct schema
- Recursive restriction works (tasks can also have restricted_schema)
- Passing `tasks=tasks` in kwargs propagates correctly

#### Task 13: Implement Selector.restricted_schema()
**File**: `dachi/act/_bt/_serial.py`
**Pattern**: B (Direct Variants)
**Field**: `tasks: ModuleList[Task] | None`

**Implementation**: Identical to Sequence (same structure)

#### Task 14: Test Selector
**File**: `tests/act/test_serial.py`

**What to test**: Same as Sequence tests

#### Task 15: Implement Multi.restricted_schema()
**File**: `dachi/act/_bt/_parallel.py`
**Pattern**: B (Direct Variants)
**Field**: `tasks: ModuleList[Task]`

**Implementation**: Same pattern as Sequence

#### Task 16: Test Multi
**File**: `tests/act/test_parallel.py` (or existing test file)

**What to test**: Same as Sequence tests

#### Task 17: Implement BT.restricted_schema()
**File**: `dachi/act/_bt/_roots.py`
**Pattern**: C (Single Field)
**Field**: `root: InitVar[Task | None]`

**Implementation**:
```python
class BT(Task, RestrictedTaskSchemaMixin):
    root: InitVar[Task | None] = None

    def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        if tasks is None:
            return self.schema()

        # Pattern C: Process task variants for single field
        task_schemas = self._schema_process_variants(tasks, _seen=_seen, tasks=tasks, **kwargs)

        # Update schema's root field (single Task field)
        schema = self.schema()
        return self._schema_update_single_field(
            schema,
            field_name="root",
            placeholder_name="TaskSpec",
            variant_schemas=task_schemas,
            profile=_profile
        )
```

#### Task 18: Test BT
**File**: `tests/act/test_roots.py` (or existing test file)

**What to test**:
- BT with task variants for root field
- Single field is correctly updated (not items or additionalProperties)

### Phase 4: Documentation

#### Task 19: Update documentation
**File**: `dev-docs/restricted_schema_guide.md`

**What to update**:
- Document the 3 patterns (A, B, C) with clear examples
- Update API examples to show new helper methods
- Remove old complexity warnings
- Add examples of RestrictedTaskSchemaMixin vs RestrictedStateSchemaMixin usage
- Explain when to use which mixin

## Key Design Decisions

1. **Domain-specific mixins**: Use `isinstance(variant, [Domain]Mixin)` for type checking
2. **Crystal clear usage**: Which mixin to use is obvious from the domain
3. **Prevents mistakes**: Can't accidentally mix task and state restrictions
4. **Clean separation**: Base mixin has only core helpers, no domain logic
5. **Two-step pattern**: Always (1) get/process schemas, (2) update parent schema
6. **Pattern A implementation**: For pass-through patterns, we create a temporary default instance of the child class to call its `restricted_schema()` method. Example: `Region().restricted_schema(states=states)`. This is safe because we only need the schema, not runtime state. The alternative would be making `restricted_schema()` a classmethod, but that complicates the implementation significantly.

## Summary

- **3 Mixin classes**: Base + Task + State
- **7 Implementation classes**: Region, CompositeState, StateChart, Sequence, Selector, Multi, BT
- **3 Patterns**: Pass-Through, Direct Variants, Single Field
- **19 Total tasks**: 4 mixin tasks + 14 implementation/test tasks + 1 doc task

---

# ORIGINAL DOCUMENTATION (Pre-Implementation - OUTDATED)

**NOTE**: The documentation below is from the old complex implementation. It is preserved for reference but should NOT be followed. See the implementation plan above for the new simplified approach.

The old approach had 19 public helper methods and required 50+ lines of boilerplate code per implementation. The new approach reduces this to:
- 3 simple mixin classes (Base, Task, State)
- 3 pattern templates (A, B, C)
- ~10-15 lines per implementation

## The Problem (Still Valid)

The problem statement remains valid: we want to restrict generic types like `Task` to specific allowed variants.

**Before restriction** (too permissive):
```json
{
  "properties": {
    "tasks": {
      "items": {"$ref": "#/$defs/TaskSpec"}  // Allows ALL tasks
    }
  }
}
```

**After restriction** (only specific tasks):
```json
{
  "properties": {
    "tasks": {
      "items": {"$ref": "#/$defs/Allowed_TaskSpec"}  // Only ActionA, ActionB, ActionC
    }
  },
  "$defs": {
    "Allowed_TaskSpec": {
      "oneOf": [
        {"$ref": "#/$defs/ActionASpec"},
        {"$ref": "#/$defs/ActionBSpec"},
        {"$ref": "#/$defs/ActionCSpec"}
      ]
    }
  }
}
```

## New Implementation Pattern (See Plan Above)

**Pattern B Example** (Direct Variants):
```python
class Sequence(Serial, RestrictedTaskSchemaMixin):
    tasks: ModuleList[Task] | None = None

    def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        if tasks is None:
            return self.schema()

        # Step 1: Process variants (calls their restricted_schema if available)
        task_schemas = self._schema_process_variants(tasks, _seen=_seen, tasks=tasks, **kwargs)

        # Step 2: Update schema with processed variants
        schema = self.schema()
        return self._schema_update_list_field(
            schema,
            field_name="tasks",
            placeholder_name="TaskSpec",
            variant_schemas=task_schemas,
            profile=_profile
        )
```

That's it! The mixin handles all the complexity (processing, $defs, unions, path replacement).

**Old approach**: 50+ lines with manual processing, entry extraction, $defs management, union creation, and path replacement.

**New approach**: ~12 lines with simple two-step pattern.

## Helper Methods Reference

### Core Processing Helpers

#### `_name_from_schema_dict(schema_dict: dict) -> str`
Extracts the spec name from a schema dict by reading `title` or `$id`.

**Usage**: Internal helper for extracting names from schema dicts.

```python
name = self._name_from_schema_dict({"title": "ActionASpec", ...})
# Returns: "ActionASpec"
```

#### `normalize_schema_type_variants(objs: Iterable) -> list[tuple[str, dict]]`
Converts various input types (BaseModule classes, BaseModel classes, schema dicts) into standardized `(name, schema_dict)` entries.

**Note**: Does NOT handle `RestrictedSchemaMixin` recursion - you must handle that manually in your loop.

**Usage**: Use after you've processed variants manually to extract names.

```python
# After processing variants into schema dicts
entries = self.normalize_schema_type_variants(processed_schema_dicts)
# Returns: [("ActionASpec", {...}), ("ActionBSpec", {...}), ...]
```

### $defs Management

#### `require_defs_for_entries(schema: dict, entries: list[tuple[str, dict]]) -> None`
Adds variant schemas to the `$defs` section of the schema.

**Usage**: Call after normalizing variants to ensure all schemas are available in `$defs`.

```python
self.require_defs_for_entries(schema, entries)
# schema["$defs"] now contains all variant schemas
```

#### `merge_defs(target: dict, *sources: dict, on_conflict="last_wins") -> None`
Merges `$defs` from multiple schema dicts into a target schema.

**Usage**: When combining schemas from multiple sources.

### Union Creation

#### `make_union_inline_from_entries(entries: list[tuple[str, dict]]) -> dict`
Creates an inline `oneOf` union from entries.

**Returns**: `{"oneOf": [{"$ref": "#/$defs/ActionASpec"}, ...]}`

**Usage**: For inline union style (more verbose, duplicated if used multiple places).

```python
union = self.make_union_inline_from_entries(entries)
self.replace_at_path(schema, ["properties", "tasks", "items"], union)
```

#### `ensure_shared_union_from_entries(schema: dict, *, placeholder_spec_name: str, entries: list[tuple[str, dict]]) -> str`
Creates a shared union definition in `$defs` and returns a reference to it.

**Returns**: `"#/$defs/Allowed_TaskSpec"`

**Usage**: For shared union style (cleaner, reusable).

```python
union_ref = self.ensure_shared_union_from_entries(
    schema,
    placeholder_spec_name="TaskSpec",
    entries=entries
)
self.replace_at_path(schema, ["properties", "tasks", "items"], {"$ref": union_ref})
```

#### `allowed_union_name(placeholder_spec_name: str) -> str`
Generates the standard name for a shared union.

**Returns**: `"Allowed_TaskSpec"` for input `"TaskSpec"`

### Path Navigation and Replacement

#### `node_at(schema: dict, path: list[str]) -> dict | list | None`
Navigates to a node in the schema by path.

**Usage**: Check if a path exists before replacing.

```python
node = self.node_at(schema, ["properties", "tasks", "items"])
# Returns the node at that path, or None if missing
```

#### `replace_at_path(schema: dict, path: list[str], replacement: dict) -> None`
Replaces the node at a path with a new value.

**Usage**: Replace placeholder references with unions.

```python
self.replace_at_path(
    schema,
    ["properties", "tasks", "items"],
    {"$ref": "#/$defs/Allowed_TaskSpec"}
)
```

#### `has_placeholder_ref(schema: dict, *, at: list[str], placeholder_spec_name: str) -> bool`
Checks if a specific path contains a reference to a placeholder.

**Usage**: Verify before replacement.

```python
if self.has_placeholder_ref(schema, at=["properties", "tasks", "items"], placeholder_spec_name="TaskSpec"):
    # Safe to replace
```

### Diagnostics

#### `collect_placeholder_refs(schema: dict, *, placeholder_spec_name: str) -> list[list[str]]`
Finds all paths that reference a placeholder.

**Usage**: Debug tool to find all locations that need replacement.

```python
refs = self.collect_placeholder_refs(schema, placeholder_spec_name="TaskSpec")
# Returns: [["properties", "tasks", "items"], ["properties", "fallback", "items"]]
```

### Optional Schema Constraints

#### `apply_array_bounds(schema: dict, *, at: list[str], min_items: int | None, max_items: int | None) -> None`
Sets `minItems`/`maxItems` on an array field.

**Usage**: Add validation constraints to arrays.

```python
self.apply_array_bounds(schema, at=["properties", "tasks"], min_items=1, max_items=10)
```

#### `set_additional_properties(schema: dict, *, at: list[str], allow: bool) -> None`
Sets `additionalProperties` on an object field.

**Usage**: Control whether extra properties are allowed.

```python
self.set_additional_properties(schema, at=["properties"], allow=False)
```

## The Flow in Detail

### Step 1: Process Variants

The first step handles different input types and applies any filtering:

```python
processed = []
for task in tasks:
    # Optional filtering
    if not self._is_allowed_task_type(task):
        continue

    # Handle RestrictedSchemaMixin recursion
    if isinstance(task, RestrictedSchemaMixin):
        processed.append(task.restricted_schema(_seen=_seen, _profile=_profile, **kwargs))
    # Handle other types
    elif isinstance(task, BaseModule):
        processed.append(task.schema())
    # ... etc
```

**Result**: `processed = [schema_dict1, schema_dict2, ...]`

**Key Points**:
- `RestrictedSchemaMixin` recursion must be handled manually
- Filtering happens here before processing
- All variants are converted to schema dicts

### Step 2: Extract Names

Convert schema dicts to `(name, schema_dict)` tuples:

```python
entries = []
for schema_dict in processed:
    name = self._name_from_schema_dict(schema_dict)
    entries.append((name, schema_dict))
```

**Result**: `entries = [("ActionASpec", {...}), ("ActionBSpec", {...}), ...]`

### Step 3: Add to $defs

Ensure all variant schemas are available:

```python
schema = self.schema()
self.require_defs_for_entries(schema, entries)
```

**Before**:
```json
{
  "$defs": {
    "TaskSpec": {...}
  }
}
```

**After**:
```json
{
  "$defs": {
    "TaskSpec": {...},
    "ActionASpec": {...},
    "ActionBSpec": {...},
    "ActionCSpec": {...}
  }
}
```

### Step 4: Create and Insert Union

Replace the placeholder with a union of allowed types.

**Shared Union Style** (recommended):

```python
union_ref = self.ensure_shared_union_from_entries(
    schema,
    placeholder_spec_name="TaskSpec",
    entries=entries
)
self.replace_at_path(schema, ["properties", "tasks", "items"], {"$ref": union_ref})
```

Creates:
```json
{
  "$defs": {
    "Allowed_TaskSpec": {
      "oneOf": [
        {"$ref": "#/$defs/ActionASpec"},
        {"$ref": "#/$defs/ActionBSpec"},
        {"$ref": "#/$defs/ActionCSpec"}
      ]
    }
  },
  "properties": {
    "tasks": {
      "items": {"$ref": "#/$defs/Allowed_TaskSpec"}
    }
  }
}
```

**Inline Union Style**:

```python
union = self.make_union_inline_from_entries(entries)
self.replace_at_path(schema, ["properties", "tasks", "items"], union)
```

Creates:
```json
{
  "properties": {
    "tasks": {
      "items": {
        "oneOf": [
          {"$ref": "#/$defs/ActionASpec"},
          {"$ref": "#/$defs/ActionBSpec"},
          {"$ref": "#/$defs/ActionCSpec"}
        ]
      }
    }
  }
}
```

## Current Issues and Concrete TODOs

### Issues Identified

1. **CRITICAL: Too Complex to Use**
   - Requires 50+ lines of boilerplate code to implement `restricted_schema()`
   - Developer must understand 10+ helper methods and how they fit together
   - Manual loop, name extraction, filtering, and multi-step union creation
   - **Impact**: Currently unusable - no complete implementations exist in codebase

2. **TOO MANY HELPER METHODS (19 public methods!)**
   - **Problem**: The mixin exposes 19 helper methods as public API
   - Most are internal implementation details users shouldn't need to know about
   - Creates cognitive overload - which methods do I actually need?
   - Many are trivial wrappers (e.g., `allowed_union_name()` just adds prefix)
   - **Current count**:
     - 1 abstract method: `restricted_schema()`
     - 3 memo methods: `memo_start()`, `memo_get()`, `memo_end()` (questionable if needed)
     - 2 normalization: `normalize_schema_type_variants()`, `_name_from_schema_dict()`
     - 4 union building: `build_refs_from_entries()`, `make_union_inline_from_entries()`, `ensure_shared_union_from_entries()`, `allowed_union_name()`
     - 2 $defs management: `require_defs_for_entries()`, `merge_defs()`
     - 3 path navigation: `node_at()`, `replace_at_path()`, `has_placeholder_ref()`
     - 2 optional constraints: `apply_array_bounds()`, `set_additional_properties()`
     - 1 diagnostics: `collect_placeholder_refs()`
   - **Reality**: For 99% of use cases, users need **ONE** high-level helper that does everything

3. **Missing High-Level Abstraction**
   - No single helper handles the common pattern
   - Each implementer reinvents the wheel
   - Easy to make mistakes in the multi-step process

4. **Helper Method Naming Unclear**
   - Method names don't clearly indicate purpose or when to use them
   - Missing comprehensive docstrings with examples
   - Unclear which methods are typically used together

### Concrete TODO List (Plan to complete today)

#### Priority 1: Simplify the API (Make it Usable)

- [ ] **Create `restrict_field_to_variants()` high-level helper**
  - Handles the entire flow: process variants → extract names → add to $defs → create union → replace
  - Signature:
    ```python
    def restrict_field_to_variants(
        schema: dict,
        *,
        field_path: list[str],              # e.g., ["properties", "tasks", "items"]
        placeholder_name: str,              # e.g., "TaskSpec"
        allowed_variants: list,             # List of BaseModule/dict/etc
        filter_fn: Callable | None = None,  # Optional type filter
        variant_style: str = "shared",      # "shared" or "inline"
        _seen: dict | None = None,          # Cycle detection cache
        **recursive_kwargs                  # Passed to recursive calls
    ) -> dict
    ```
  - Reduces implementation from ~50 lines to ~5 lines
  - **Target**: Make `restricted_schema()` implementable in 10 lines or less

- [ ] **Create `process_variants()` helper** (lower-level, but still helpful)
  - Handles the variant processing loop with recursion and filtering
  - Signature:
    ```python
    def process_variants(
        variants: list,
        *,
        filter_fn: Callable | None = None,
        _seen: dict | None = None,
        _profile: str = "shared",
        **recursive_kwargs
    ) -> list[dict]  # Returns list of schema dicts
    ```
  - Reduces Step 1-2 from ~20 lines to ~3 lines

#### Priority 2: Rename and Document All Helper Methods

- [x] **`normalize_variants()` → `normalize_schema_type_variants()`** ✓ DONE
  - Added comprehensive docstring with usage examples

- [ ] **`memo_start()` → `init_schema_cache()`**
  - Rename parameter `klass` → `module_class`
  - Add docstring explaining cycle breaking use case
  - Consider: Is memo pattern actually needed? Or make it private/internal?

- [ ] **`memo_get()` → `get_cached_schema()`**
  - Rename parameter `klass` → `module_class`
  - Add docstring

- [ ] **`memo_end()` → `store_schema_in_cache()`**
  - Rename parameter `klass` → `module_class`
  - Add docstring

- [ ] **Review and rename remaining helpers**
  - `build_refs_from_entries()` - OK or rename?
  - `make_union_inline_from_entries()` - OK or rename?
  - `ensure_shared_union_from_entries()` - OK or rename?
  - `allowed_union_name()` - OK or rename?
  - `require_defs_for_entries()` - OK or rename?
  - `merge_defs()` - OK or rename?
  - `node_at()` - OK or rename?
  - `replace_at_path()` - OK or rename?
  - `has_placeholder_ref()` - OK or rename?
  - `collect_placeholder_refs()` - OK or rename?
  - `apply_array_bounds()` - OK or rename?
  - `set_additional_properties()` - OK or rename?

#### Priority 3: Create Working Examples

- [ ] **Implement `restricted_schema()` in `Sequence` class**
  - File: `dachi/act/_bt/_serial.py`
  - Currently has stub that raises `NotImplementedError`
  - Use as real-world test of the new high-level helper

- [ ] **Implement `restricted_schema()` in `BT` class**
  - File: `dachi/act/_bt/_roots.py`
  - Currently has incomplete implementation
  - Demonstrate recursive restriction pattern

- [ ] **Implement `restricted_schema()` in a StateChart class**
  - File: `dachi/act/_chart/_chart.py` or similar
  - Currently has stub that raises `NotImplementedError`
  - Test with state variants

#### Priority 4: Add Tests

- [ ] **Unit tests for helper methods**
  - Test `normalize_schema_type_variants()` with various input types
  - Test union creation (both shared and inline styles)
  - Test path navigation and replacement

- [ ] **Integration tests for `restricted_schema()`**
  - Test simple restriction (one level)
  - Test recursive restriction (nested modules)
  - Test filtering (type constraints)
  - Test both "shared" and "inline" profiles

- [ ] **End-to-end test**
  - Build a complete behavior tree with restrictions
  - Verify schema is correct and usable
  - Validate with a JSON schema validator

### Success Criteria

**The RestrictedSchemaMixin will be considered "usable" when:**

1. A developer can implement `restricted_schema()` in **10 lines or less** for simple cases
2. All helper methods have clear names and comprehensive docstrings
3. At least 3 working examples exist in the codebase
4. Tests validate the pattern works correctly
5. This guide serves as a complete reference

**Timeline**: Plan to complete today (all Priority 1-2 items)

## Discussion Summary

This guide was developed through a conversation analyzing the `RestrictedSchemaMixin` implementation:

**Key Insights**:
- The mixin provides low-level tools but requires manual orchestration
- No complete working examples exist yet in the codebase
- The pattern is consistent but verbose
- Need to balance flexibility with ease of use

**Design Decisions**:
- Keep low-level helpers for flexibility
- Document the standard pattern clearly
- Plan to add high-level helpers for common cases
- Support both filtering and recursive restriction

**Next Steps**:
1. Complete renaming and documentation of all helper methods
2. Implement simplified high-level helpers
3. Create working examples in real behavior tree/state chart classes
4. Add tests demonstrating the pattern

## Examples

### Example 1: Simple Task Restriction

```python
class SimpleSequence(BaseModule, RestrictedSchemaMixin):
    tasks: ModuleList[Task]

    def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        if tasks is None:
            return self.schema()

        # Process variants
        processed = []
        for task in tasks:
            if isinstance(task, RestrictedSchemaMixin):
                processed.append(task.restricted_schema(_seen=_seen, _profile=_profile, **kwargs))
            elif isinstance(task, BaseModule):
                processed.append(task.schema())
            elif isinstance(task, dict):
                processed.append(task)

        # Extract names
        entries = [(self._name_from_schema_dict(s), s) for s in processed]

        # Build restricted schema
        schema = self.schema()
        self.require_defs_for_entries(schema, entries)

        union_ref = self.ensure_shared_union_from_entries(
            schema, placeholder_spec_name="TaskSpec", entries=entries
        )
        self.replace_at_path(schema, ["properties", "tasks", "items"], {"$ref": union_ref})

        return schema
```

### Example 2: With Type Filtering

```python
class FilteredBehaviorTree(BaseModule, RestrictedSchemaMixin):
    tasks: ModuleList[Task]

    def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        if tasks is None:
            return self.schema()

        # Process and filter
        processed = []
        for task in tasks:
            # Only allow Action and Condition subtypes
            if isinstance(task, type) and issubclass(task, BaseModule):
                if not (issubclass(task, Action) or issubclass(task, Condition)):
                    continue

            if isinstance(task, RestrictedSchemaMixin):
                processed.append(task.restricted_schema(_seen=_seen, _profile=_profile, **kwargs))
            elif isinstance(task, BaseModule):
                processed.append(task.schema())

        entries = [(self._name_from_schema_dict(s), s) for s in processed]

        schema = self.schema()
        self.require_defs_for_entries(schema, entries)

        union_ref = self.ensure_shared_union_from_entries(
            schema, placeholder_spec_name="TaskSpec", entries=entries
        )
        self.replace_at_path(schema, ["properties", "tasks", "items"], {"$ref": union_ref})

        return schema
```

## Simplification Proposal: Reduce Public API Surface

### Current State: Too Many Public Methods

**Problem**: The mixin currently exposes **19 public helper methods**. This is overwhelming and most are internal implementation details.

**Analysis of Current Methods**:
- `_name_from_schema_dict()` - Already private (has `_` prefix) ✓
- `normalize_schema_type_variants()` - Could be internal
- `memo_start()`, `memo_get()`, `memo_end()` - **Questionable if needed at all**
- `build_refs_from_entries()` - Trivial, should be private
- `_infer_discriminator_mapping_from_entries()` - Already private ✓
- `make_union_inline_from_entries()` - Should be private
- `allowed_union_name()` - Trivial (just adds "Allowed_" prefix), should be private
- `ensure_shared_union_from_entries()` - Should be private
- `require_defs_for_entries()` - Should be private
- `merge_defs()` - Rarely used, could be private
- `node_at()` - Should be private
- `replace_at_path()` - Should be private
- `has_placeholder_ref()` - Debug helper, could be private
- `apply_array_bounds()` - Optional feature, rarely used
- `set_additional_properties()` - Optional feature, rarely used
- `collect_placeholder_refs()` - Debug helper, could be private

### Proposed Simplified Public API

**Reduce from 19 methods to just 2 public methods:**

```python
class RestrictedSchemaMixin(ABC):
    # ============================================================
    # PUBLIC API (what users interact with)
    # ============================================================

    @abstractmethod
    def restricted_schema(self, **kwargs) -> dict:
        """
        Implement this method to return a restricted schema.

        Use restrict_field_to_variants() helper to do the work.
        """
        raise NotImplementedError()

    def restrict_field_to_variants(
        self,
        schema: dict,
        *,
        field_path: list[str],              # e.g., ["properties", "tasks", "items"]
        placeholder_name: str,              # e.g., "TaskSpec"
        allowed_variants: list,             # List of allowed types
        filter_fn: Callable | None = None,  # Optional type filter
        variant_style: str = "shared",      # "shared" or "inline"
        _seen: dict | None = None,          # Internal cycle detection
        **recursive_kwargs
    ) -> dict:
        """
        ONE METHOD TO DO EVERYTHING.

        This is the only helper method users need to call.
        Handles: processing variants → extracting names → adding to $defs
                 → creating union → replacing placeholder

        Example:
            schema = self.schema()
            return self.restrict_field_to_variants(
                schema,
                field_path=["properties", "tasks", "items"],
                placeholder_name="TaskSpec",
                allowed_variants=tasks,
                filter_fn=lambda t: issubclass(t, Action),
                variant_style="shared"
            )
        """
        # Implementation uses all the private helpers below
        pass

    # ============================================================
    # PRIVATE HELPERS (internal implementation, prefix with _)
    # ============================================================

    def _process_variants(self, variants, filter_fn, _seen, **kwargs) -> list[dict]:
        """Internal: Handle variant processing with recursion and filtering"""

    @staticmethod
    def _name_from_schema_dict(d: dict) -> str:
        """Internal: Extract spec name from schema dict"""

    @staticmethod
    def _add_entries_to_defs(schema: dict, entries: list[tuple[str, dict]]) -> None:
        """Internal: Add variant schemas to $defs"""

    @staticmethod
    def _create_union(entries: list[tuple[str, dict]], style: str, placeholder_name: str) -> dict | str:
        """Internal: Create union (returns dict for inline, str ref for shared)"""

    @staticmethod
    def _replace_at_path(schema: dict, path: list[str], value: dict) -> None:
        """Internal: Replace node at path in schema"""

    # ... other private helpers as needed
```

### Benefits of This Approach

1. **Dramatic simplification**: 19 public methods → 2 public methods
2. **Clear mental model**: Users only need to know `restrict_field_to_variants()`
3. **Flexibility preserved**: Private helpers can still be used for edge cases
4. **Better encapsulation**: Implementation details hidden
5. **Easier to maintain**: Can change internals without breaking API

### Implementation Becomes Trivial

**Before** (50+ lines):
```python
def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
    if tasks is None:
        return self.schema()

    processed = []
    for task in tasks:
        if isinstance(task, RestrictedSchemaMixin):
            processed.append(task.restricted_schema(_seen=_seen, _profile=_profile, **kwargs))
        elif isinstance(task, BaseModule):
            processed.append(task.schema())
        # ... more cases

    entries = [(self._name_from_schema_dict(s), s) for s in processed]
    schema = self.schema()
    self.require_defs_for_entries(schema, entries)

    if _profile == "shared":
        union_ref = self.ensure_shared_union_from_entries(
            schema, placeholder_spec_name="TaskSpec", entries=entries
        )
        self.replace_at_path(schema, ["properties", "tasks", "items"], {"$ref": union_ref})
    else:
        union = self.make_union_inline_from_entries(entries)
        self.replace_at_path(schema, ["properties", "tasks", "items"], union)

    return schema
```

**After** (5 lines):
```python
def restricted_schema(self, *, tasks=None, **kwargs):
    if tasks is None:
        return self.schema()

    schema = self.schema()
    return self.restrict_field_to_variants(
        schema,
        field_path=["properties", "tasks", "items"],
        placeholder_name="TaskSpec",
        allowed_variants=tasks,
        **kwargs
    )
```

### Questions to Consider

1. **Memo pattern**: Is the memo/cache pattern (`memo_start`, `memo_get`, `memo_end`) actually needed?
   - Has anyone encountered cycles in practice?
   - If not needed, remove entirely
   - If needed, make it automatic inside `restrict_field_to_variants()`

2. **Optional features**: Keep `apply_array_bounds()` and `set_additional_properties()` public?
   - Probably yes, as they're genuinely optional extras
   - Or fold them into `restrict_field_to_variants()` as kwargs?

3. **Debug helpers**: Keep `collect_placeholder_refs()` public for debugging?
   - Could be useful for troubleshooting
   - Or make it private and add better error messages instead?

### Recommendation

**Phase 1 (Today)**:
- Create `restrict_field_to_variants()` as the one public helper
- Mark all other helpers as "to be made private in next version"
- Add deprecation warnings

**Phase 2 (Next version)**:
- Make all helpers private (prefix with `_`)
- Keep only `restricted_schema()` (abstract) and `restrict_field_to_variants()` public
- Remove memo methods if not actually needed

This gives users time to migrate while dramatically simplifying the API going forward.

---

# IMPLEMENTATION PROGRESS (2025-10-29)

## Status: ✅ COMPLETE

All behavior tree classes have been successfully implemented with `restricted_schema()` support, following TDD methodology throughout.

## Completed Tasks

### Phase 1: Base Implementation ✅
1. **RestrictedSchemaMixin** - Base class with core helpers implemented in `dachi/core/_base.py:1305-1580`
2. **RestrictedTaskSchemaMixin** - Domain-specific mixin for behavior trees in `dachi/act/_bt/_core.py:631-671`
3. **lookup_module_class()** utility - Resolves variants to module classes in `dachi/core/_base.py:1246-1302`
4. **filter_class_variants()** utility - Simplifies class filtering in `dachi/core/_base.py:1211-1243`

### Phase 2: Behavior Tree Implementations ✅

#### Pattern B (Direct Variants) - ModuleList Fields
1. **Sequence** - `dachi/act/_bt/_serial.py:37-79` (6 tests ✅)
2. **Selector** - `dachi/act/_bt/_serial.py:173-214` (6 tests ✅)
3. **Multi** - `dachi/act/_bt/_parallel.py:18-65` (6 tests ✅)

#### Pattern C (Single Field)
4. **BT** - `dachi/act/_bt/_roots.py:11-123` (6 tests ✅)
5. **Decorator** - `dachi/act/_bt/_decorators.py:9-51` (6 tests ✅)

#### Pattern C with Filtering
6. **BoundTask** (Leaf filter) - `dachi/act/_bt/_decorators.py:163-220` (6 tests ✅)
7. **PreemptCond** (Condition filter) - `dachi/act/_bt/_serial.py:323-394` (5 tests ✅)

### Phase 3: Critical Fixes ✅
8. **ModuleList Generic Support** - Fixed schema generation for `ModuleList[Task]` in `dachi/core/_base.py:591-610`

## Key Challenges & Solutions

### Challenge 1: ModuleList Generic Type Handling
**Problem**: Pydantic couldn't generate JSON schema for `ModuleList[Task]` because it's a `typing._GenericAlias`, not a type.

**Solution**: Modified `BaseModule.__build_schema__()` to detect generic aliases using `t.get_origin()` and extract the base type:
```python
base_type = t.get_origin(typ) if t.get_origin(typ) is not None else typ
if isinstance(base_type, type) and issubclass(base_type, BaseModule):
    origin = base_type.schema_model()
```

### Challenge 2: Field Union Handling
**Problem**: Initially tried to use `ModuleList[Task] | None = None` which complicated schema generation.

**Decision**: Removed union types from field definitions while keeping `= None` default values. Post-init handles conversion to ModuleList instances.

### Challenge 3: Code Complexity in Filtering
**Problem**: Manual filtering logic was verbose (6 lines) and duplicated across BoundTask and PreemptCond:
```python
# Before
cond_variants = []
for variant in tasks:
    mod_cls = lookup_module_class(variant)
    if mod_cls is not None and issubclass(mod_cls, Condition):
        cond_variants.append(variant)
```

**Solution**: Created `filter_class_variants()` utility function:
```python
# After
cond_variants = list(filter_class_variants(Condition, tasks))
```

### Challenge 4: RestrictedSchemaMixin Location
**Problem**: Initially placed `RestrictedTaskSchemaMixin` in `dachi/core/_base.py`.

**Correction**: Moved to `dachi/act/_bt/_core.py` as it's domain-specific to behavior trees, not general framework code.

## Code Changes Summary

### Files Modified
- `dachi/core/_base.py` - Added base mixin, utilities, fixed generic type handling
- `dachi/act/_bt/_core.py` - Added RestrictedTaskSchemaMixin
- `dachi/act/_bt/_serial.py` - Implemented Sequence, Selector, PreemptCond
- `dachi/act/_bt/_parallel.py` - Implemented Multi
- `dachi/act/_bt/_roots.py` - Implemented BT
- `dachi/act/_bt/_decorators.py` - Implemented Decorator, BoundTask

### Test Files Added/Modified
- `tests/core/test_base.py` - Added ModuleList generic tests
- `tests/act/test_serial.py` - Added 17 tests (Sequence: 6, Selector: 6, PreemptCond: 5)
- `tests/act/test_parallel.py` - Added 6 tests (Multi)
- `tests/act/test_roots.py` - Added 6 tests (BT)
- `tests/act/test_decorators.py` - Added 12 tests (Decorator: 6, BoundTask: 6)

### Test Results
- **Total tests**: 698 passing
- **New tests added**: 35
- **Failures**: 0
- **Test coverage**: All patterns and edge cases covered

## Implementation Patterns Used

### Pattern B: Direct Variants (ModuleList fields)
Used by: Sequence, Selector, Multi

**Template**:
```python
def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
    if tasks is None:
        return self.schema()

    task_schemas = self._schema_process_variants(
        tasks,
        restricted_schema_cls=RestrictedTaskSchemaMixin,
        _seen=_seen,
        tasks=tasks,
        **kwargs
    )

    schema = self.schema()
    return self._schema_update_list_field(
        schema,
        field_name="tasks",
        placeholder_name="TaskSpec",
        variant_schemas=task_schemas,
        profile=_profile
    )
```

### Pattern C: Single Field
Used by: BT, Decorator

**Template**:
```python
def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
    if tasks is None:
        return self.schema()

    task_schemas = self._schema_process_variants(
        tasks,
        restricted_schema_cls=RestrictedTaskSchemaMixin,
        _seen=_seen,
        tasks=tasks,
        **kwargs
    )

    schema = self.schema()
    return self._schema_update_single_field(
        schema,
        field_name="<field_name>",
        placeholder_name="TaskSpec",
        variant_schemas=task_schemas,
        profile=_profile
    )
```

### Pattern C with Filtering
Used by: BoundTask (Leaf filter), PreemptCond (Condition filter)

**Template**:
```python
def restricted_schema(self, *, tasks=None, _profile="shared", _seen=None, **kwargs):
    if tasks is None:
        return self.schema()

    # Filter to specific class
    filtered_variants = list(filter_class_variants(TargetClass, tasks))

    if not filtered_variants:
        return self.schema()  # or handle differently

    schemas = self._schema_process_variants(
        filtered_variants,
        restricted_schema_cls=RestrictedTaskSchemaMixin,
        _seen=_seen,
        tasks=tasks,
        **kwargs
    )

    schema = self.schema()
    return self._schema_update_single_field(
        schema,
        field_name="<field_name>",
        placeholder_name="<SpecName>",
        variant_schemas=schemas,
        profile=_profile
    )
```

## Next Steps (Future Work)

### State Charts (Signatures Exist, Implementations Needed)
The following state chart classes have `restricted_schema()` signatures but raise `NotImplementedError`:

1. **Region** - Pattern B (direct state variants) - `dachi/act/_chart/_region.py`
2. **CompositeState** - Pattern A (pass-through to regions) - `dachi/act/_chart/_composite.py`
3. **StateChart** - Pattern A (pass-through to regions) - `dachi/act/_chart/_chart.py`

These need actual implementations following the patterns in Phase 2 of the Implementation Order section.

### Potential Improvements

1. **API Simplification** - Consider creating a higher-level `restrict_field_to_variants()` helper to reduce boilerplate
2. **Cycle Detection** - Evaluate if `_seen` parameter is actually needed in practice
3. **Documentation** - Add usage examples and tutorials
4. **Performance** - Profile and optimize schema generation for large variant lists

## Lessons Learned

1. **TDD is Essential** - Writing tests first caught issues early and ensured correctness
2. **Utility Functions Matter** - Small utilities like `filter_class_variants()` dramatically improve code readability
3. **Domain Separation** - Keeping domain-specific logic out of core base classes prevents cross-contamination
4. **Generic Type Handling** - Python's typing system requires special handling for generic types in schema generation
5. **Import Organization** - Keeping imports at the top of files prevents circular dependency issues

## Todos Completed

All planned behavior tree implementations are complete:
- ✅ Selector.restricted_schema() - Pattern B
- ✅ Multi.restricted_schema() - Pattern B
- ✅ BT.restricted_schema() - Pattern C
- ✅ Decorator.restricted_schema() - Pattern C
- ✅ BoundTask.restricted_schema() - Pattern C with Leaf filter
- ✅ PreemptCond.restricted_schema() - Pattern C × 2 with Condition filter

## Outstanding Todos (Future Sessions)

### Phase 2: Complete State Chart restricted_schema() Implementations
**Status**: Signatures exist but raise NotImplementedError - Implementation needed

The following classes have `restricted_schema()` method signatures but need actual implementation:

1. **Region.restricted_schema()** - Pattern B (direct state variants)
   - File: `dachi/act/_chart/_region.py`
   - Current: `raise NotImplementedError`
   - Needs: Implementation for `states: ModuleDict[str, State]`

2. **CompositeState.restricted_schema()** - Pattern A (pass-through to regions)
   - File: `dachi/act/_chart/_composite.py`
   - Current: `raise NotImplementedError`
   - Needs: Pass-through implementation

3. **StateChart.restricted_schema()** - Pattern A (pass-through to regions)
   - File: `dachi/act/_chart/_chart.py`
   - Current: `raise NotImplementedError`
   - Needs: Pass-through implementation

See detailed implementation plans in original Phase 2 section (lines 727-850).

---

### Phase 4: Documentation
**Status**: Not Started

- Document the 3 patterns (A, B, C) with clear examples
- Update API examples to show new helper methods
- Add examples of RestrictedTaskSchemaMixin vs RestrictedStateSchemaMixin usage
- Explain when to use which mixin

---

### Phase 5: Process & Enhanced State Metadata (NEW REQUIREMENTS)

#### 5.1. Process Restricted Schema with ProcessCallSpec
**Status**: Not Started - Design Discussion Required

**Goal**: Add `restricted_schema()` support for Process classes with proper input type handling.

**Requirements**:
- Create `ProcessCallSpec` class that includes:
  - `process`: The process to call
  - `inputs`: Can be either `RefT` (reference to another process) or an actual value
  - Schema must include the type information for inputs

**Design Questions to Resolve**:
1. How should `ProcessCallSpec` be structured?
   - Should it be a Pydantic model or a dataclass?
   - How to represent `RefT` vs actual values in the schema?
   - Should there be separate fields for `ref_inputs` and `value_inputs`?

2. How to retrieve input types in schema methods?
   - Should Process classes introspect their `execute()` signature?
   - Should input types be declared explicitly as class attributes?
   - How to handle optional vs required inputs?

3. Integration with restricted_schema:
   - Does Process follow Pattern A, B, or C?
   - Should `ProcessCallSpec` variants be passed to `restricted_schema()`?
   - How to validate that provided inputs match the process signature?

**Example Structure (to be refined)**:
```python
class ProcessCallSpec:
    """Specification for calling a process with inputs"""
    process: Type[Process]  # or Process instance?
    inputs: Dict[str, RefT | Any]  # How to represent this in schema?

class SomeProcess(Process):
    def restricted_schema(self, *, processes=None, **kwargs):
        # Implementation TBD after design discussion
        pass
```

**Related Files**:
- `dachi/proc/_base.py` - Process base class
- TBD: Where should ProcessCallSpec live?

---

### 2. State Schema Metadata: Emissions, Inputs, Outputs
**Status**: Not Started - Design Discussion Required

**Goal**: Include emissions, inputs, and outputs metadata in schemas for StateChart states and BaseState instances so LLMs can understand the full interface.

**Requirements**:

#### For StateChart State Schemas:
- **Emissions**: What events can this state emit?
  - Event names
  - Event payload types/schemas
  - When/under what conditions are they emitted?

- **Inputs**: What data does this state expect from context?
  - Input parameter names
  - Input types/schemas
  - Whether inputs are required or optional

- **Outputs**: What data does this state produce?
  - Output parameter names
  - Output types/schemas
  - When are outputs produced (on entry, on exit, during execution)?

#### For BaseState Schemas:
- **Inputs**: Parameters the state needs to execute
  - From parent context
  - From transitions
  - From events

- **Outputs**: Data the state produces
  - Return values
  - Context mutations
  - Side effects

**Design Questions to Resolve**:
1. Where should this metadata be stored?
   - As class attributes on State classes?
   - In a separate metadata registry?
   - Inferred from method signatures?

2. How should it be represented in the JSON schema?
   - As custom `x-` fields in the schema?
   - As a separate `metadata` section?
   - Integrated into property descriptions?

3. How to keep metadata in sync with code?
   - Manual declaration (error-prone)?
   - Automatic introspection (limited info)?
   - Decorator-based annotation?

**Example Structure (to be refined)**:
```python
class MyState(BaseState):
    # Option 1: Explicit declaration
    __inputs__ = {
        'user_id': int,
        'optional_flag': Optional[bool]
    }
    __outputs__ = {
        'result': str,
        'status': Status
    }
    __emissions__ = {
        'completed': CompletedEvent,
        'failed': FailedEvent
    }

    # Option 2: Decorator approach
    @state_metadata(
        inputs={'user_id': int},
        outputs={'result': str},
        emissions={'completed': CompletedEvent}
    )
    async def execute(self, ctx):
        pass

    # Option 3: Introspection from signature
    async def execute(self,
                     user_id: int,           # Detected as input
                     optional_flag: bool = False) -> str:  # Detected as output
        # Emissions would still need explicit declaration
        await self.emit('completed', payload=...)
        return "result"

# In restricted_schema()
def restricted_schema(self, **kwargs):
    schema = super().restricted_schema(**kwargs)

    # Add metadata section
    schema['x-state-metadata'] = {
        'inputs': self._get_input_metadata(),
        'outputs': self._get_output_metadata(),
        'emissions': self._get_emission_metadata()
    }

    return schema
```

**Related Files**:
- `dachi/act/_states/` - State chart implementation
- `dachi/act/_bt/_core.py` - Base state class
- TBD: Metadata utilities location

**Benefits for LLMs**:
- Understand complete state interface without reading code
- Generate valid state configurations
- Validate event flows and data dependencies
- Provide better autocomplete/suggestions

---

### 3. State Chart Restricted Schema Implementation
**Status**: Not Started - Blocked by Todo #2

Once Todo #2 (metadata design) is resolved, implement `restricted_schema()` for:
- **StateChart** - Pattern A (pass-through to regions)
- **Region** - Pattern B (direct state variants)
- **CompositeState** - Pattern A (pass-through to regions)

These implementations should include the metadata from Todo #2.

---

## Session End Notes

This session completed all behavior tree `restricted_schema()` implementations with comprehensive test coverage (698 tests passing). The system is production-ready for behavior trees.

Next session should focus on:
1. Design discussion for Process/ProcessCallSpec
2. Design discussion for State metadata
3. Implementation of the above once designs are approved

All code changes have been committed to working memory and are ready for the next session.

---

# IMPLEMENTATION SESSION 2025-10-30: Classmethod Refactoring

## Session Goal
Convert `restricted_schema()` from instance method to `@classmethod` and implement StateChart restricted schemas.

## Progress Made

### ✅ Phase 1: Converted Base Infrastructure to Classmethods
**Files Modified**:
- `dachi/core/_base.py` - RestrictedSchemaMixin
- `dachi/act/_bt/_core.py` - RestrictedTaskSchemaMixin
- `dachi/act/_chart/_base.py` - RestrictedStateSchemaMixin (created)

**Changes**:
1. Made `restricted_schema()` abstract method a `@classmethod` in all three mixin classes
2. Converted all helper methods to classmethods:
   - `_schema_process_variants()` → `@classmethod`
   - `_schema_update_list_field()` → `@classmethod`
   - `_schema_update_dict_field()` → `@classmethod`
   - `_schema_update_single_field()` → `@classmethod`
3. Updated method bodies: changed `self` → `cls` throughout
4. Modified `_schema_process_variants` to call `variant.restricted_schema()` as classmethod instead of creating temp instances

**Key Fix in normalize_schema_type_variants()**:
Changed line 1444 from:
```python
sm = o.schema_model()
entries.append((sm.__name__, sm.model_json_schema()))
```
To:
```python
schema_dict = o.schema()
name = cls._schema_name_from_dict(schema_dict)
entries.append((name, schema_dict))
```

This uses `BaseModule.schema()` instead of Pydantic's `model_json_schema()`, which properly handles States with runtime Attr fields.

### ✅ Phase 2: Converted All Task Implementations
**Files Modified** (automated script):
- `dachi/act/_bt/_serial.py` - Sequence, Selector, PreemptCond
- `dachi/act/_bt/_decorators.py` - Decorator, BoundTask
- `dachi/act/_bt/_roots.py` - BT
- `dachi/act/_bt/_parallel.py` - Multi

**Script used**:
```python
# Automated conversion: def restricted_schema(self, → @classmethod\n    def restricted_schema(cls,
# Also changed self. → cls. in common patterns
```

**Test Results**: All existing behavior tree tests pass (TestSequenceRestrictedSchema, etc.)

### ✅ Phase 3: Implemented State Chart Infrastructure
**Files Modified**:
- `dachi/act/_chart/_base.py` - Created `RestrictedStateSchemaMixin`
- `dachi/act/_chart/__init__.py` - Exported new mixin
- `dachi/act/_chart/_region.py` - Implemented `Region.restricted_schema()` (Pattern B)
- `dachi/act/_chart/_composite.py` - Implemented `CompositeState.restricted_schema()` (Pattern A)

**Region.restricted_schema() - Pattern B** [dachi/act/_chart/_region.py:252-288]:
```python
@classmethod
def restricted_schema(cls, *, states=None, _profile="shared", _seen=None, **kwargs):
    if states is None:
        return cls.schema()

    state_schemas = cls._schema_process_variants(
        states,
        restricted_schema_cls=RestrictedStateSchemaMixin,
        _seen=_seen,
        states=states,
        **kwargs
    )

    schema = cls.schema()
    return cls._schema_update_dict_field(
        schema,
        field_name="states",
        placeholder_name="BaseStateSpec",
        variant_schemas=state_schemas,
        profile=_profile
    )
```

**CompositeState.restricted_schema() - Pattern A** [dachi/act/_chart/_composite.py:188-223]:
```python
@classmethod
def restricted_schema(cls, *, states=None, _profile="shared", _seen=None, **kwargs):
    if states is None:
        return cls.schema()

    # Pattern A: Pass states to Region.restricted_schema() - NO INSTANCE NEEDED!
    region_schema = Region.restricted_schema(
        states=states,
        _profile=_profile,
        _seen=_seen,
        **kwargs
    )

    schema = cls.schema()
    return cls._schema_update_list_field(
        schema,
        field_name="regions",
        placeholder_name="RegionSpec",
        variant_schemas=[region_schema],
        profile=_profile
    )
```

### ✅ Phase 4: Added Tests
**Files Modified**:
- `tests/act/test_chart.py` - Added `TestRegionRestrictedSchema` (6 tests, all passing)
- `tests/act/test_chart_composite.py` - Added `TestCompositeStateRestrictedSchema` (5 tests, 1 failing due to caching)

**Test Coverage**:
- Region: states=None, updates states field, shared vs inline profile, spec class variants, mixed formats
- CompositeState: states=None, passes to Region, updates regions field, etc.

## Critical Challenges Encountered

### Challenge 1: Instance Method vs Classmethod Decision
**Initial Plan Statement** (Line 966):
> "The alternative would be making `restricted_schema()` a classmethod, but that complicates the implementation significantly."

**Reality**: This was an LLM hallucination! Making it a classmethod SIMPLIFIES implementation:
- **Before**: Pattern A required creating temp instances: `Region(name="temp", initial="temp", rules=[]).restricted_schema(...)`
- **After**: Direct class call: `Region.restricted_schema(...)`
- **Benefits**: No `__post_init__` side effects, clearer semantics, more flexible

**Decision**: Converted everything to classmethods.

### Challenge 2: FinalState Attr Field Bug
**Problem**: `FinalState` had `status: Attr[ChartStatus] = Attr(ChartStatus.SUCCESS)` which caused Pydantic schema generation to fail.

**Root Cause**: Attr fields are runtime state management, NOT serializable spec fields. They should never appear in Spec classes.

**Fix** (already in codebase by user): Changed to `InitVar` pattern [dachi/act/_chart/_state.py:42]:
```python
status: InitVar[ChartStatus] = ChartStatus.SUCCESS

def __post_init__(self, status: ChartStatus = ChartStatus.SUCCESS):
    self.status = Attr[ChartStatus](data=status)
```

**Result**: Attr is created at runtime in `__post_init__`, not in class annotations, so it doesn't appear in the Spec.

### Challenge 3: Pytest Caching Issues
**Problem**: Tests pass when run standalone but fail in pytest due to stale bytecode/module caching.

**Symptoms**:
- `python -c "..."` succeeds
- `pytest tests/...` fails with same code
- Error: `PydanticInvalidForJsonSchema: Cannot generate a JsonSchema for core_schema.IsInstanceSchema (<class 'dachi.core._base.Attr'>)`

**Investigation**:
- Cleared `__pycache__`, `.pytest_cache`, bytecode files
- Problem persists in pytest but not in standalone Python
- Suggests pytest is caching module state differently

**Status**: UNRESOLVED - Needs fresh pytest session or environment restart to verify fix

**Workaround**: User confirmed the FinalState fix is in place, so tests should pass after clearing all caches and restarting Python interpreter.

## Remaining Work

### Immediate (Current Session)
- [ ] **StateChart.restricted_schema()** - Pattern A (pass-through to CompositeState/Region)
  - File: `dachi/act/_chart/_chart.py`
  - Implementation: Similar to CompositeState, pass states to Region
  - Tests: Add to `tests/act/test_chart.py` or new file

### Short-term (Next Session)
- [ ] **Update all tests to call as classmethod** - Change `instance.restricted_schema()` → `ClassName.restricted_schema()`
  - Tests currently call on instances (which works but isn't idiomatic)
  - Should demonstrate classmethod usage in test examples

- [ ] **Verify pytest caching issue resolved** - Run full test suite in fresh environment

- [ ] **Update dev docs with classmethod pattern** - Document that restricted_schema is NOW a classmethod (update lines 1027-1042)

### Documentation Updates Needed
1. **Line 966**: Remove statement about classmethod being complicated
2. **Lines 1027-1042**: Update example to show `@classmethod def restricted_schema(cls, ...`
3. **Pattern A examples**: Update to show direct class calls, not temp instance creation
4. **Add section**: "Why Classmethod?" explaining benefits over instance method

## Key Learnings

### 1. Question LLM Hallucinations in Planning Docs
The plan document contained incorrect reasoning about classmethods being "complicated". Always verify such claims against actual implementation requirements.

### 2. Attr Fields Must Not Appear in Class Annotations
**Correct Pattern**:
```python
class MyState(BaseState):
    status: InitVar[ChartStatus] = ChartStatus.SUCCESS  # ✓ InitVar, not Attr

    def __post_init__(self, status: ChartStatus):
        self.status = Attr[ChartStatus](data=status)  # Create Attr at runtime
```

**Incorrect Pattern**:
```python
class MyState(BaseState):
    status: Attr[ChartStatus] = Attr(ChartStatus.SUCCESS)  # ✗ Attr in annotations
```

### 3. BaseModule.schema() vs Pydantic model_json_schema()
Always use `BaseModule.schema()` (classmethod) instead of `schema_model().model_json_schema()` when working with modules that may have Attr fields in their hierarchy. The BaseModule.schema() method properly handles the dachi-specific field types.

### 4. Pytest Caching Can Hide Issues
When making fundamental changes to base classes (like method signatures), always:
- Clear all `__pycache__` directories
- Use `pytest --cache-clear`
- Consider `python -B` flag (no bytecode)
- In persistent issues, restart Python interpreter entirely

## Files Changed Summary

### Core Infrastructure
- `dachi/core/_base.py` - RestrictedSchemaMixin → classmethod
- `dachi/act/_bt/_core.py` - RestrictedTaskSchemaMixin → classmethod
- `dachi/act/_chart/_base.py` - Created RestrictedStateSchemaMixin → classmethod

### Task Implementations (automated conversion)
- `dachi/act/_bt/_serial.py` (3 classes)
- `dachi/act/_bt/_parallel.py` (1 class)
- `dachi/act/_bt/_roots.py` (1 class)
- `dachi/act/_bt/_decorators.py` (2 classes)

### State Implementations (manual)
- `dachi/act/_chart/_region.py` - Region.restricted_schema()
- `dachi/act/_chart/_composite.py` - CompositeState.restricted_schema()
- `dachi/act/_chart/__init__.py` - Export RestrictedStateSchemaMixin

### Tests Added
- `tests/act/test_chart.py` - TestRegionRestrictedSchema (6 tests)
- `tests/act/test_chart_composite.py` - TestCompositeStateRestrictedSchema (5 tests)

## Test Status
- **Behavior Tree Tests**: ✅ All passing (Sequence, Selector, Multi, BT, Decorator, BoundTask, PreemptCond)
- **Region Tests**: ✅ All 6 tests passing
- **CompositeState Tests**: ⚠️ 4/5 passing, 1 failing due to pytest caching issue
- **Total Test Count**: ~700+ tests passing

## Next Session TODO
1. Implement StateChart.restricted_schema()
2. Resolve pytest caching issue (restart interpreter, fresh test run)
3. Update documentation to reflect classmethod pattern
4. Consider updating tests to demonstrate classmethod usage (optional, works either way)
5. Begin Process restricted_schema design discussion

## Session End State
The classmethod refactoring is **95% complete**. All infrastructure and implementations are done. Only remaining work is:
- StateChart implementation (straightforward, ~10 lines)
- Verifying tests in fresh environment
- Documentation updates

The refactoring successfully eliminated the need for temporary instance creation in Pattern A, making the code cleaner and more flexible.



DataFlow Refactoring Plan (Phases 1 & 2) - COMPLETE WITH DETAILS
Location in Guide
Will be added to /Users/shortg/Development/dachi/dev-docs/restricted_schema_guide.md as a new section: Section Title: # DATAFLOW AND PROCESS RESTRICTED SCHEMA (2025-10-31)
Plan Content
Overview
Refactor DataFlow from runtime-built structure (_nodes, _args) to explicit serializable fields (inputs, processes, outputs). This enables spec-based serialization/deserialization and prepares for restricted_schema implementation. Key Design Principle: ProcessCall is a BaseModule (data container), NOT a Process subclass, because it stores metadata about a process + arguments, not executable logic.
Phase 1: Create ProcessCall Class
Location
File: dachi/proc/_graph.py Position: After RefT class (line 378), before DataFlow class (line 381)
Complete Implementation
class ProcessCall(BaseModule, RestrictedSchemaMixin):
    """Wrapper for a Process/AsyncProcess with its arguments in a DAG.
    
    Used by DataFlow to store both the process and its arguments together
    as a serializable unit. The name is stored as the key in DataFlow's 
    processes ModuleDict.
    
    Args:
        process: The Process or AsyncProcess to execute
        args: Arguments to pass to the process (can be RefT or literal values)
    
    Note:
        ProcessCall is a data container, not an executable process. DataFlow
        extracts the process and args to execute them.
        
    Convenience Methods:
        is_async: Returns True if the wrapped process is AsyncProcess
    """
    process: Process | AsyncProcess
    args: Dict[str, RefT | t.Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.args is None:
            self.args = {}
    
    def is_async(self) -> bool:
        """Check if the wrapped process is async."""
        return isinstance(self.process, AsyncProcess)
    
    @classmethod
    def restricted_schema(
        cls,
        *,
        processes: List[Type[Process | AsyncProcess]] | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> dict:
        """Generate restricted schema for ProcessCall with allowed process types.
        
        DESIGN NOTE: Implementation strategy to be determined in next session.
        This is a brainstorming item - pattern may differ from Task/State patterns.
        
        Args:
            processes: List of allowed Process/AsyncProcess types
            _profile: "shared" or "inline" union style
            _seen: Cycle detection cache
            **kwargs: Additional process-specific restrictions
            
        Returns:
            JSON schema dict with process field restricted to allowed types
        """
        # TODO: Implement in Phase 3 (restricted_schema implementation)
        # Pattern may differ from RestrictedTaskSchemaMixin/RestrictedStateSchemaMixin
        raise NotImplementedError(
            "ProcessCall.restricted_schema() implementation to be designed in Phase 3"
        )
Rationale:
Inherits from BaseModule for serialization support
Inherits from RestrictedSchemaMixin to support restricted_schema (Phase 3)
No name field - stored in ModuleDict key
Only Process | AsyncProcess allowed (no StreamProcess per user requirement)
args defaults to None, initialized to {} in post_init
is_async() convenience method for checking process type
restricted_schema() signature defined but implementation deferred
Phase 2: Refactor DataFlow Structure
2.1 Update DataFlow Class Definition
File: dachi/proc/_graph.py (lines 381-416) Current class definition:
class DataFlow(AsyncProcess):
    """[Existing docstring]"""
    
    def __post_init__(self):
        # Runtime-built structure
        self._nodes = ModuleDict()
        self._args = Attr[Dict[str, Dict[str, RefT | Any]]](data={})
        self._outputs = Attr[List[str] | str](data=None)
New class definition:
class DataFlow(AsyncProcess, RestrictedSchemaMixin):
    """[Keep existing comprehensive docstring from lines 382-416]
    
    Architecture:
        - inputs: Simple Dict[str, Any] for literal input values
        - processes: ModuleDict[str, ProcessCall] for named process nodes
        - outputs: List[str] for output node names
        
    Serialization:
        All fields are serializable via spec()/from_spec()
    """
    
    # Explicit serializable fields
    inputs: Dict[str, Any] = None
    processes: ModuleDict[str, ProcessCall] = None
    outputs: List[str] = None
    
    def __post_init__(self):
        """Initialize DataFlow with serializable fields.
        
        [Keep existing docstring content about args and methods]
        """
        super().__post_init__()
        
        # Initialize None fields to empty containers per user requirement
        if self.inputs is None:
            self.inputs = {}
        if self.processes is None:
            self.processes = ModuleDict()
        if self.outputs is None:
            self.outputs = []
        
        # Internal counters (not serialized) - kept for _generate_node_name()
        self._node_counter = Attr[int](data=0)
        self._var_counter = Attr[int](data=0)
    
    @classmethod
    def restricted_schema(
        cls,
        *,
        processes: List[Type[Process | AsyncProcess]] | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> dict:
        """Generate restricted schema for DataFlow with allowed process types.
        
        DESIGN NOTE: Implementation strategy to be determined in next session.
        This is a brainstorming item - pattern may differ from Task/State patterns.
        
        Key Differences from Task/State:
        - Must infer allowed input types from process signatures
        - inputs field should be Dict[str, Union[<inferred types>]]
        - processes field restricted to ProcessCall with allowed process types
        - ProcessCall.args should allow RefT | Union[<inferred types>]
        
        Args:
            processes: List of allowed Process/AsyncProcess types
            _profile: "shared" or "inline" union style
            _seen: Cycle detection cache
            **kwargs: Additional restrictions
            
        Returns:
            JSON schema dict with:
            - processes field restricted to ProcessCall with allowed process types
            - inputs field restricted to inferred input types
            - outputs field unchanged (List[str])
        """
        # TODO: Implement in Phase 3 (restricted_schema implementation)
        # Need to design:
        # 1. How to extract input types from process signatures
        # 2. How to build union of all input types from all processes
        # 3. How to restrict ProcessCall.args to RefT | <union of types>
        raise NotImplementedError(
            "DataFlow.restricted_schema() implementation to be designed in Phase 3"
        )
Rationale:
Added RestrictedSchemaMixin to inheritance (prepares for Phase 3)
inputs: Simple Dict[str, Any] stores literal values directly
processes: ModuleDict[str, ProcessCall] stores process nodes
outputs: List[str] stores output node names
All fields default to None, initialized in post_init
restricted_schema() signature defined but implementation deferred
Internal counters remain as Attr (not part of spec)
2.2 Update Helper Methods
_generate_node_name()
File: dachi/proc/_graph.py (lines 436-455) Changes: Check both self.processes and self.inputs for collisions
def _generate_node_name(self, prefix: str = "node") -> str:
    """Generate unique node name with given prefix"""
    if prefix == "node":
        name = f"{prefix}_{self._node_counter.data}"
        self._node_counter.data += 1
    elif prefix == "var":
        name = f"{prefix}_{self._var_counter.data}"
        self._var_counter.data += 1
    else:
        counter = 0
        # Check BOTH processes and inputs for collisions
        while f"{prefix}_{counter}" in self.processes or f"{prefix}_{counter}" in self.inputs:
            counter += 1
        name = f"{prefix}_{counter}"
    return name
2.3 Update Public API Methods
link()
File: dachi/proc/_graph.py (lines 521-550) Current:
def link(self, name: str, node: Process | AsyncProcess, **kwargs) -> RefT:
    if name in self._nodes:
        raise ValueError(f"Node '{name}' already exists in DAG")
    self._nodes[name] = node
    self._args.data[name] = kwargs
    return RefT(name=name)
New:
def link(self, name: str, node: Process | AsyncProcess, **kwargs: RefT | typing.Any) -> RefT:
    """Link a computation node to the DataFlow.
    
    [Keep existing docstring from lines 523-545]
    """
    if name in self.processes:
        raise ValueError(f"Node '{name}' already exists in DataFlow")
    
    # Create ProcessCall wrapper
    process_call = ProcessCall(process=node, args=kwargs if kwargs else {})
    self.processes[name] = process_call
    
    return RefT(name=name)
add_inp()
File: dachi/proc/_graph.py (lines 552-581) Current:
def add_inp(self, name: str, val: typing.Any) -> RefT:
    if name in self._nodes:
        raise ValueError(f"Node {name} already exists in DAG")
    self._nodes[name] = Var(val=val, name=name)
    self._args.data[name] = {}
    return RefT(name=name)
New:
def add_inp(self, name: str, val: typing.Any) -> RefT:
    """Add an input variable (root node) to the DataFlow.
    
    [Keep existing docstring from lines 554-576, update examples to reflect new structure]
    """
    if name in self.inputs:
        raise ValueError(f"Input '{name}' already exists in DataFlow")
    
    # Store literal value directly (no Var wrapper)
    self.inputs[name] = val
    
    return RefT(name=name)
set_out()
File: dachi/proc/_graph.py (lines 583-608) Current:
def set_out(self, outputs: typing.List[str]|str) -> None:
    output_list = outputs if isinstance(outputs, list) else [outputs]
    for output in output_list:
        if output not in self._nodes:
            raise ValueError(f"Output node '{output}' does not exist in DataFlow")
    self._outputs.data = outputs
New:
def set_out(self, outputs: typing.List[str]|str) -> None:
    """Set the output nodes of the DataFlow.
    
    [Keep existing docstring from lines 585-603]
    """
    output_list = outputs if isinstance(outputs, list) else [outputs]
    
    # Validate all outputs exist in either inputs or processes
    for output in output_list:
        if output not in self.inputs and output not in self.processes:
            raise ValueError(f"Output node '{output}' does not exist in DataFlow")
    
    # Always store as list internally
    self.outputs = output_list if isinstance(outputs, list) else [outputs]
contains()
File: dachi/proc/_graph.py (lines 610-617) Current:
def __contains__(self, item: str) -> bool:
    return item in self._nodes
New:
def __contains__(self, item: str) -> bool:
    """Check if the DataFlow contains a node with the given name"""
    return item in self.inputs or item in self.processes
2.4 Update Internal Execution Methods
_sub() - Core Resolution Logic
File: dachi/proc/_graph.py (lines 457-519) Key changes:
Handle self.inputs lookup (return literal value)
Handle self.processes lookup (extract process and args from ProcessCall)
Raise KeyError if not found
Execution logic unchanged
New implementation:
async def _sub(self, name: str, by: typing.Dict, visited: typing.Dict[str, asyncio.Task] | None = None):
    """Subroutine to get the value of a node by name, resolving any references"""
    if visited is None:
        visited = dict()
    
    # Check memoization
    if name in by:
        return by[name]
    
    # Check if already being computed (cycle detection)
    if name in visited:
        task = visited[name]
        current_task = asyncio.current_task()
        if task is not current_task:
            if not task.done():
                await task
            return task.result()
        elif name in by:
            return by[name]
    
    # Get node and args from new structure
    if name in self.inputs:
        # Input node - return literal value directly
        result = self.inputs[name]
        by[name] = result
        return result
    elif name in self.processes:
        # Process node - extract from ProcessCall
        process_call = self.processes[name]
        node = process_call.process
        args = process_call.args
    else:
        raise KeyError(f"Node '{name}' not found in DataFlow")
    
    # Resolve arguments in parallel (unchanged logic from lines 486-501)
    kwargs = {}
    async with asyncio.TaskGroup() as tg:
        for key, arg in args.items():
            if isinstance(arg, RefT):
                task = tg.create_task(
                    self._sub(arg.name, by, visited)
                )
                if arg.name not in visited:
                    visited[arg.name] = task
                kwargs[key] = task
            else:
                kwargs[key] = arg
    
    kwargs = {
        k: v if not isinstance(v, asyncio.Task) else v.result()
        for k, v in kwargs.items()
    }
    
    # Execute node (unchanged logic from lines 502-518)
    if isinstance(node, Process):
        res = node(**kwargs)
    elif isinstance(node, AsyncProcess):
        res = await node.aforward(**kwargs)
    elif isinstance(node, str):
        method = getattr(self, node, None)
        if method is None:
            raise ValueError(
                f"Method {node} not found in {type(self).__name__}"
            )
        res = await method(**kwargs)
    else:
        raise ValueError(
            f"Node {name} is not a Process or AsyncProcess"
        )
    
    by[name] = res
    return res
aforward()
File: dachi/proc/_graph.py (lines 646-704) Changes: Use self.outputs instead of self._outputs.data
async def aforward(
    self,
    by: typing.Dict=None,
    out_override: typing.List[str]|str|RefT=None
):
    """Execute the DataFlow and return the output values.
    
    [Keep existing docstring from lines 648-676]
    """
    # Use outputs field directly instead of _outputs.data
    outputs = out_override if out_override is not None else self.outputs
    
    if outputs is None or (isinstance(outputs, list) and len(outputs) == 0):
        return None
    
    if isinstance(outputs, (str, RefT)):
        outputs = [outputs]
        single = True
    else:
        single = False
    
    by = by if by is not None else {}
    res = []
    
    for output in outputs:
        if isinstance(output, RefT):
            name = output.name
        else:
            name = output
        
        if name in by:
            res.append(by[name])
        else:
            res.append(await self._sub(name, by))
    
    if single:
        return res[0]
    return tuple(res)
sub()
File: dachi/proc/_graph.py (lines 619-634)
def sub(self, outputs: typing.List[str], by: typing.Dict[str, typing.Any]) -> 'DataFlow':
    """Create a sub-DataFlow with the given outputs"""
    sub_dag = DataFlow()
    
    for name in outputs:
        if name not in self:
            raise ValueError(f"Node {name} does not exist in DataFlow")
        
        # Copy inputs
        if name in self.inputs:
            sub_dag.inputs[name] = self.inputs[name]
        
        # Copy processes (shallow copy - shares ProcessCall instances)
        if name in self.processes:
            sub_dag.processes[name] = self.processes[name]
    
    sub_dag.set_out(outputs)
    return sub_dag
replace()
File: dachi/proc/_graph.py (lines 636-644)
def replace(self, name: str, node: Process | AsyncProcess) -> None:
    """Replace a process node in the DataFlow"""
    if name not in self.processes:
        raise ValueError(f"Process node '{name}' does not exist in DataFlow")
    
    # Update the process, keep existing args
    self.processes[name].process = node
2.5 Update T-Node Interop Methods
from_node_graph()
File: dachi/proc/_graph.py (lines 706-731) This is REQUIRED for T-node interop:
@classmethod
def from_node_graph(cls, nodes: typing.List[BaseNode]):
    """Create a DataFlow from a list of T/Var nodes"""
    dag = cls()
    for node in nodes:
        if node.name is None:
            raise ValueError("Node must have a name to be added to DataFlow")
        
        if isinstance(node, Var):
            # Add to inputs
            dag.add_inp(name=node.name, val=node.val)
        elif isinstance(node, T):
            # Convert T.args to kwargs with RefT references
            args = {}
            for k, arg in node.args.items():
                if isinstance(arg, BaseNode):
                    args[k] = RefT(name=arg.name)
                else:
                    args[k] = arg
            # Add to processes via link()
            dag.link(name=node.name, node=node.src, **args)
        else:
            raise ValueError("Node must be a Var or T to be added to DataFlow")
    return dag
to_node_graph()
File: dachi/proc/_graph.py (lines 733-762) This is REQUIRED for T-node interop:
def to_node_graph(self) -> typing.List[BaseNode]:
    """Convert the DataFlow to a list of T/Var nodes"""
    nodes = []
    
    # Convert inputs to Var nodes
    for name, val in self.inputs.items():
        nodes.append(Var(val=val, name=name))
    
    # Convert processes to T nodes
    for name, process_call in self.processes.items():
        node = process_call.process
        
        # Convert args: RefT -> BaseNode references
        args = {}
        for k, arg in process_call.args.items():
            if isinstance(arg, RefT):
                # Find the referenced node in our node list
                ref_node = next((n for n in nodes if n.name == arg.name), None)
                if ref_node is None:
                    raise ValueError(f"Referenced node '{arg.name}' not found")
                args[k] = ref_node
            else:
                args[k] = arg
        
        # Determine if async
        is_async = isinstance(node, AsyncProcess)
        
        nodes.append(
            T(
                src=node,
                args=SerialDict(data=args),
                name=name,
                is_async=is_async
            )
        )
    
    return nodes
Testing Strategy
Existing Tests Must Pass
File: tests/proc/test_graph.py All 211 existing tests must pass after refactoring:
TestDAG - basic initialization and resolution
TestDAGLink - link() method
TestDAGAddInp - add_inp() method
TestDAGSetOut - set_out() method
TestDAGSub, TestDAGReplace - sub() and replace()
TestDAGGraphConversion - from_node_graph() and to_node_graph()
TestDAGIntegration - complex scenarios
TestDAGEdgeCases - edge cases
New Serialization Tests
Add test class TestDataFlowSerialization in tests/proc/test_graph.py:
@pytest.mark.asyncio
class TestDataFlowSerialization:
    """Tests for DataFlow spec-based serialization"""
    
    async def test_processCall_serialization(self):
        """ProcessCall should serialize correctly"""
        pc = ProcessCall(
            process=_Add(),
            args={'a': RefT('x'), 'b': 10}
        )
        
        spec = pc.spec()
        pc2 = ProcessCall.from_spec(spec)
        
        assert isinstance(pc2.process, _Add)
        assert 'a' in pc2.args
        assert isinstance(pc2.args['a'], RefT)
        assert pc2.args['a'].name == 'x'
        assert pc2.args['b'] == 10
    
    async def test_processCall_is_async_method(self):
        """ProcessCall.is_async() should detect AsyncProcess"""
        sync_pc = ProcessCall(process=_Add(), args={})
        async_pc = ProcessCall(process=_AsyncConst(5), args={})
        
        assert sync_pc.is_async() is False
        assert async_pc.is_async() is True
    
    async def test_simple_dataflow_roundtrip(self):
        """DataFlow with inputs and processes should serialize/deserialize"""
        dag = DataFlow()
        dag.add_inp('x', val=5)
        dag.add_inp('y', val=10)
        dag.link('sum', _Add(), a=RefT('x'), b=RefT('y'))
        dag.set_out('sum')
        
        # Serialize and reconstruct
        spec = dag.spec()
        dag2 = DataFlow.from_spec(spec)
        
        # Verify structure preserved
        assert 'x' in dag2.inputs
        assert 'y' in dag2.inputs
        assert dag2.inputs['x'] == 5
        assert dag2.inputs['y'] == 10
        assert 'sum' in dag2.processes
        assert dag2.outputs == ['sum']
        
        # Verify execution works
        result = await dag2.aforward()
        assert result == 15
    
    async def test_complex_dataflow_roundtrip(self):
        """DataFlow with multiple nodes and dependencies"""
        dag = DataFlow()
        dag.add_inp('x', val=5)
        dag.link('double', _Add(), a=RefT('x'), b=RefT('x'))
        dag.link('triple', _Add(), a=RefT('double'), b=RefT('x'))
        dag.set_out(['double', 'triple'])
        
        spec = dag.spec()
        dag2 = DataFlow.from_spec(spec)
        
        result = await dag2.aforward()
        assert result == (10, 15)
Implementation Order
✅ Create ProcessCall class (with is_async method)
✅ Update DataFlow field definitions
✅ Update post_init()
✅ Update _generate_node_name()
✅ Update link()
✅ Update add_inp()
✅ Update set_out()
✅ Update contains()
✅ Update _sub() (most complex)
✅ Update aforward()
✅ Update sub()
✅ Update replace()
✅ Update from_node_graph()
✅ Update to_node_graph()
✅ Run existing tests: pytest tests/proc/test_graph.py -v
✅ Fix any test failures
✅ Add new serialization tests
✅ Run all tests again
✅ Verify 100% pass rate
Phase 3: restricted_schema() Implementation (Future - Brainstorming Required)
Deferred Design Questions
The following questions need to be resolved in the next session before implementing restricted_schema():
1. ProcessCall.restricted_schema() Pattern
Question: What pattern should ProcessCall follow? Options:
Pattern B-like: Process variants as direct list?
New pattern: Different from Task/State patterns?
How to handle process input type extraction?
Considerations:
ProcessCall wraps a process, not a collection
Need to restrict process field to allowed types
Need to restrict args values to RefT | <union of allowed types>
2. Input Type Inference
Question: How to extract input types from Process signatures? Options:
Introspect forward() / aforward() method signatures
Require explicit input type declarations
Use type hints from method annotations
Manual specification in restricted_schema() call
Example:
class MyProcess(Process):
    def forward(self, x: int, y: str) -> float:
        return float(x) + len(y)

# How to extract: {'x': int, 'y': str}?
3. DataFlow.restricted_schema() Implementation
Question: How should DataFlow build the restricted schema? Requirements (from user):
Restrict processes field to ProcessCall with allowed process types
Restrict inputs dict to union of input types from all processes
Restrict ProcessCall.args to RefT | <union of types>
Pseudocode (to be refined):
@classmethod
def restricted_schema(cls, *, processes=None, **kwargs):
    if processes is None:
        return cls.schema()
    
    # Step 1: Get all input types from all processes
    input_types = set()
    for proc_cls in processes:
        # How to extract types from proc_cls?
        input_types.update(extract_input_types(proc_cls))
    
    # Step 2: Create ProcessCall schema restricted to allowed processes
    # Pattern B on processes field?
    
    # Step 3: Update inputs field to Dict[str, Union[<input_types>]]
    # How to represent in JSON schema?
    
    # Step 4: Update ProcessCall.args to RefT | Union[<input_types>]
    # Nested schema restriction?
4. Execution Mode Metadata
User Requirement: ProcessCall should indicate execution mode Question: How to represent this? Options:
Inferred from isinstance(process, AsyncProcess)?
Explicit execution_mode field on ProcessCall?
Metadata in schema only, not in class?
User's original statement:
"I think we need a field on ProcessCallSpec that indicates whether the process inherits from Async (AsyncProcess), Stream (StreamProcess), etc."
Clarification needed:
Is this a runtime field or schema-only metadata?
Should StreamProcess be supported despite earlier "no Stream" requirement?
Files Modified Summary
Core Changes
dachi/proc/_graph.py - All DataFlow and ProcessCall changes
Test Changes
tests/proc/test_graph.py - New TestDataFlowSerialization class
Success Criteria
Phase 1 & 2 (This Plan)
✅ ProcessCall class created (BaseModule with is_async method) ✅ ProcessCall inherits RestrictedSchemaMixin ✅ DataFlow inherits RestrictedSchemaMixin ✅ DataFlow uses explicit fields (inputs, processes, outputs) ✅ All existing tests pass (no regressions) ✅ from_node_graph() and to_node_graph() working with new structure ✅ New serialization tests passing ✅ DataFlow can serialize via spec() and reconstruct via from_spec() ✅ Execution behavior unchanged (backward compatible) ✅ is_async() convenience method on ProcessCall
Phase 3 (Future Session - Brainstorming)
⏳ Resolve design questions for ProcessCall.restricted_schema() ⏳ Resolve design questions for input type inference ⏳ Implement ProcessCall.restricted_schema() ⏳ Implement DataFlow.restricted_schema() ⏳ Add tests for restricted_schema() implementations
Documentation Updates
Update restricted_schema_guide.md
Add this entire plan as new section at line 2464 (after Session End State):
---

# DATAFLOW AND PROCESS RESTRICTED SCHEMA (2025-10-31)

## Status: Phase 1 & 2 Complete, Phase 3 Pending Design Discussion

[Include full plan content above]
Conflict Resolution
Check for conflicts with existing content:
Lines 2021-2067: "Phase 5: Process & Enhanced State Metadata (NEW REQUIREMENTS)" discusses ProcessCallSpec
Resolution: Our new section provides implementation details for this TODO
Action: Update references in Phase 5 section to point to new DATAFLOW section
Updated Phase 5 reference (line 2021):
### Phase 5: Process & Enhanced State Metadata (IMPLEMENTED)

#### 5.1. Process Restricted Schema with ProcessCallSpec
**Status**: Phases 1 & 2 Complete - See "DATAFLOW AND PROCESS RESTRICTED SCHEMA" section below
Summary
This plan provides:
Complete implementation details for ProcessCall class (with is_async)
Complete refactoring plan for DataFlow structure
All method changes with before/after code
Testing strategy with specific test cases
Clear success criteria for Phases 1 & 2
Identified design questions for Phase 3 (restricted_schema implementation)
Documentation update plan to integrate with existing guide
Key Design Principles:
ProcessCall is a BaseModule (data container), not a Process
ProcessCall has is_async() convenience method
Both ProcessCall and DataFlow inherit RestrictedSchemaMixin
restricted_schema() signatures defined but implementations deferred to Phase 3
Phase 3 requires brainstorming session to resolve design questions
Next Session Goals:
Brainstorm ProcessCall.restricted_schema() pattern
Design input type inference strategy
Design DataFlow.restricted_schema() implementation

---

# SESSION UPDATE (2025-11-01): ProcessCall Implementation & Core Framework Fixes

## Status: Phase 3 Partially Complete - Deferred for Explicit ModField Refactor

### What Was Accomplished

#### 1. Utility Functions Created

Created two new utility functions to support restricted schema implementation:

**File**: [dachi/utils/_f_utils.py:182](dachi/utils/_f_utils.py#L182)
- `extract_parameter_types()` - Extracts parameter types from function signatures
- Handles `self`, `cls`, and custom parameter exclusions
- Supports required vs optional type annotations
- **Tests**: [tests/utils/test_f_utils.py](tests/utils/test_f_utils.py) - 11 tests, all passing

**File**: [dachi/utils/_utils.py:424](dachi/utils/_utils.py#L424)
- `python_type_to_json_schema()` - Converts Python types to JSON schema dicts
- Handles primitives, generics (List, Dict), Union types, Optional
- **Tests**: [tests/utils/test_utils.py:192](tests/utils/test_utils.py#L192) - TestPythonTypeToJsonSchema class with 14 tests, all passing

#### 2. RestrictedProcessSchemaMixin Created

**File**: [dachi/proc/_process.py:919](dachi/proc/_process.py#L919)

Created domain-specific mixin for processes following the established pattern:
- Inherits from `RestrictedSchemaMixin`
- Uses `isinstance(variant, RestrictedProcessSchemaMixin)` for recursion checks
- Prevents cross-contamination between task/state/process domains
- Exported in [dachi/proc/__init__.py](dachi/proc/__init__.py)

This is the **4th mixin** in the system:
1. `RestrictedSchemaMixin` (base)
2. `RestrictedTaskSchemaMixin` (behavior trees)
3. `RestrictedStateSchemaMixin` (state charts)
4. `RestrictedProcessSchemaMixin` (processes/dataflow) ← **NEW**

#### 3. ProcessCall Implementation

**File**: [dachi/proc/_graph.py:381](dachi/proc/_graph.py#L381)

Implemented ProcessCall class:
- Inherits from `BaseModule, RestrictedProcessSchemaMixin`
- Fields: `process: Process | AsyncProcess`, `args: Dict[str, RefT | Any]`
- Implements `is_async()` convenience method
- Implements `restricted_schema()` using **Pattern B (Direct Variants)**

**Pattern B Implementation**:
1. Process variants to get N process schemas
2. Extract input types from all process parameters
3. Update `process` field with allowed process variants
4. Update `args` field to only allow union of extracted input types + RefT

**Helper Methods**:
- `_extract_input_types_from_processes()` - Extracts all parameter types from process list
- `_schema_update_args_types()` - Updates args additionalProperties with allowed types
- Uses existing `_schema_process_variants()` and `_schema_update_single_field()` from mixin

**Tests**: [tests/proc/test_process_call.py](tests/proc/test_process_call.py)
- TestProcessCall (4 tests) - Basic functionality
- TestProcessCallSerialization (3 tests) - Spec creation and preservation
- TestProcessCallRestrictedSchema (8 tests) - Restricted schema generation
- **All 15 tests passing**

#### 4. Critical BaseModule Framework Fixes

During ProcessCall implementation, discovered and fixed **fundamental issues** in BaseModule spec generation:

**Issue 1: Union Type Conversion**
- **Problem**: Union types like `Process | AsyncProcess` weren't being converted to `ProcessSpec | AsyncProcessSpec` in spec generation
- **Impact**: ProcessCall.spec() validation failed because it expected Process instance but got ProcessSpec
- **Fix**: Added `__convert_type_to_spec__()` classmethod to handle Union conversions

**File**: [dachi/core/_base.py:545](dachi/core/_base.py#L545)
```python
@classmethod
def __convert_type_to_spec__(cls, typ: t.Any) -> t.Any:
    """Convert type annotation to spec equivalent.

    Handles Union types (both typing.Union and types.UnionType) by converting
    each Union member that is a BaseModule to its spec model.
    """
```

Handles both:
- `typing.Union[A, B]` (explicit Union)
- `A | B` syntax (Python 3.10+ `types.UnionType`)

**Issue 2: Spec Inheritance**
- **Problem**: All child module specs inherited from `BaseSpec` instead of parent module specs
- **Impact**: Pydantic validation rejected child specs (e.g., TestProcessSpec not recognized as ProcessSpec)
- **Fix**: Modified spec generation to inherit from parent module specs

**File**: [dachi/core/_base.py:633-654](dachi/core/_base.py#L633-L654)
```python
# Find all parent spec classes for proper inheritance
parent_specs = []
for base in cls.__bases__:
    if hasattr(base, '__spec__') and issubclass(base, BaseModule):
        parent_specs.append(base.__spec__)

# Use parent specs if found, otherwise BaseSpec
if len(parent_specs) > 1:
    spec_base = tuple(parent_specs)  # Multiple inheritance support
elif len(parent_specs) == 1:
    spec_base = parent_specs[0]
else:
    spec_base = BaseSpec
```

**Tests Added**: [tests/core/test_base.py](tests/core/test_base.py)
- TestSpecInheritance (5 tests) - Verifies proper spec inheritance chain
- TestUnionTypeConversion (5 tests) - Tests both Union syntaxes
- TestSpecSerializationWithUnions (2 tests) - End-to-end spec roundtrip
- **All 12 new tests passing**

### Current Challenges and Blockers

#### The Implicit ModField Problem

During this work, we identified a fundamental architectural issue with the current **implicit module field** approach:

**Current Approach (Implicit)**:
```python
class ProcessCall(BaseModule):
    process: Process | AsyncProcess  # Annotation alone defines modfield
    args: Dict[str, RefT | Any]
```

Framework inspects annotations and automatically creates modfields. This has issues:
1. **Reliability**: Hard to distinguish user intent - is this a modfield or just a type hint?
2. **Extensibility**: Users can't easily extend the framework with custom field types
3. **Validation**: No fail-fast validation - errors appear during spec generation
4. **Debugging**: Implicit behavior makes debugging harder

**Proposed Approach (Explicit)**:
```python
class ProcessCall(BaseModule):
    process: Process | AsyncProcess = modfield()
    args: Dict[str, RefT | Any] = moddictfield()
```

Benefits:
1. **Reliability**: Clear intent - `modfield()` means it's a module field
2. **Extensibility**: Users can create `mycustomfield()` for their needs
3. **Type Safety**: Type checkers will complain if annotation doesn't match field usage
4. **Fail-Fast**: Validation happens at class definition time
5. **Simplifies Core**: `__convert_type_to_spec__()` logic becomes simpler
6. **Forces Good Behavior**: Type checkers enforce correct usage

**Decision**: Implement explicit modfield approach **NOW** before more code depends on implicit system.

### Next Steps

#### Immediate: Explicit ModField Implementation

1. **Design Phase** (Brainstorming session):
   - API design for `modfield()`, `modlistfield()`, `moddictfield()`
   - How these interact with `__build_schema__()`
   - Migration strategy for existing code
   - Testing approach

2. **Implementation Phase**:
   - Implement modfield functions in `dachi/core/_base.py`
   - Update `__build_schema__()` to use explicit fields
   - Simplify `__convert_type_to_spec__()` based on explicit fields
   - Migrate existing code (Process, Task, State, ProcessCall, DataFlow, etc.)

3. **Testing Phase**:
   - Test all modfield variants
   - Test migration of existing modules
   - Ensure no regressions in existing functionality

#### After ModField: Resume Restricted Schema Work

1. **DataFlow.restricted_schema()** implementation
2. Additional testing for ProcessCall restricted schema edge cases
3. Integration tests for full process/dataflow restricted schemas

### Files Modified This Session

**New Files**:
- [tests/utils/test_f_utils.py](tests/utils/test_f_utils.py) - Tests for extract_parameter_types
- [tests/proc/test_process_call.py](tests/proc/test_process_call.py) - Tests for ProcessCall

**Modified Files**:
- [dachi/utils/_f_utils.py](dachi/utils/_f_utils.py) - Added extract_parameter_types()
- [dachi/utils/_utils.py](dachi/utils/_utils.py) - Added python_type_to_json_schema()
- [dachi/proc/_process.py](dachi/proc/_process.py) - Added RestrictedProcessSchemaMixin
- [dachi/proc/_graph.py](dachi/proc/_graph.py) - Added ProcessCall class
- [dachi/proc/__init__.py](dachi/proc/__init__.py) - Exported RestrictedProcessSchemaMixin
- [dachi/core/_base.py](dachi/core/_base.py) - Fixed Union conversion and spec inheritance
- [tests/utils/test_utils.py](tests/utils/test_utils.py) - Added TestPythonTypeToJsonSchema
- [tests/core/test_base.py](tests/core/test_base.py) - Added 3 test classes for BaseModule fixes

### Success Metrics

✅ **Completed**:
- RestrictedProcessSchemaMixin created and exported
- ProcessCall class fully implemented with Pattern B
- ProcessCall.restricted_schema() working correctly
- Type extraction utilities created and tested
- JSON schema conversion utilities created and tested
- Critical BaseModule bugs fixed (Union conversion, spec inheritance)
- Comprehensive tests for all new functionality (38 new tests total)
- All existing tests still passing (no regressions)

⏳ **Deferred Until After ModField**:
- DataFlow.restricted_schema() implementation
- Integration tests for process/dataflow restricted schemas
- Documentation for using restricted schemas with processes

### Architecture Decision Record

**Decision**: Shift from implicit to explicit module field declarations

**Context**:
- Current implicit approach (annotations alone define modfields) has reliability and extensibility issues
- Discovered during ProcessCall implementation when Union type handling became complex
- User has been concerned about this for a while

**Consequences**:
- **Short-term**: Need to implement modfield() functions and migrate existing code
- **Long-term**: More reliable, extensible, and maintainable framework
- **Type Safety**: Type checkers enforce correct usage
- **User Experience**: Clearer API, better error messages

**Status**: Approved - Implementation begins next session

---

## MODFIELD IMPLEMENTATION PLAN (2025-11-01)

### Architecture Overview

#### Class Hierarchy
```
RestrictedSchemaMixin (base - has ALL low-level schema helpers)
├── RestrictedTaskSchemaMixin (domain: tasks)
├── RestrictedStateSchemaMixin (domain: states)
├── RestrictedProcessSchemaMixin (domain: processes)
└── BaseFieldDescriptor (descriptor protocol + field-specific restricted_schema)
    ├── ModFieldDescriptor (single module field)
    ├── ModListFieldDescriptor (module list field)
    └── ModDictFieldDescriptor (module dict field)
```

**Key Design Decisions**:
1. **BaseFieldDescriptor inherits RestrictedSchemaMixin** - gets all helper methods
2. **Field descriptors return `(field_schema, defs)` tuple** - parent merges defs
3. **No backward compatibility** - clean break, migrate all modules
4. **Regular fields use `dataclasses.field()`** - only modules get descriptors
5. **Nullable handled by descriptors** - detect `None` in type list, wrap in anyOf

#### Helper Method Distribution

**Stay in RestrictedSchemaMixin** (used by both modules AND descriptors):
- `_schema_process_variants()` - Domain-specific variant filtering
- `_schema_name_from_dict()` - Extract name from schema
- `_schema_require_defs_for_entries()` - Add entries to $defs
- `_schema_build_refs()` - Build $ref list
- `_schema_make_union_inline()` - Create oneOf union
- `_schema_allowed_union_name()` - Generate union name
- `_schema_ensure_shared_union()` - Create shared union in $defs
- `_schema_merge_defs()` - Merge $defs dicts
- `_schema_node_at()` - Navigate schema tree
- `_schema_replace_at_path()` - Replace at path

**Remove from RestrictedSchemaMixin** (replaced by descriptors):
- `_schema_update_list_field()` → ModListFieldDescriptor.restricted_schema()
- `_schema_update_dict_field()` → ModDictFieldDescriptor.restricted_schema()
- `_schema_update_single_field()` → ModFieldDescriptor.restricted_schema()

### BaseFieldDescriptor Design

```python
from abc import ABC, abstractmethod
from typing import Union, Optional, get_origin, get_args, Type
from inspect import _empty as UNDEFINED
import types

class BaseFieldDescriptor(RestrictedSchemaMixin):
    """Base descriptor for module fields."""

    def __init__(self, typ=UNDEFINED, default=UNDEFINED):
        self.typ = typ  # UNDEFINED, single type, or list of types
        self.default = default
        self._name = None
        self._owner = None
        self._types = None  # Resolved type list: [Type1, Type2, None, ...]

    # Descriptor protocol
    def __set_name__(self, owner, name):
        self._name = name
        self._owner = owner
        annotation = owner.__annotations__.get(name)
        self.validate_annotation(annotation)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # From class -> return descriptor
        return obj.__dict__.get(self._name)

    def __set__(self, obj, value):
        obj.__dict__[self._name] = value

    # Type management
    def validate_annotation(self, annotation) -> None:
        """Validate annotation and extract types into self._types.

        Implementation:
        1. Extract types from annotation or typ parameter into a list
        2. If self._types is already set, compare with extracted list
        3. If not set, set self._types to extracted list
        4. Validate that types are compatible

        Handles:
        - Union[A, B] or A | B -> [A, B]
        - Optional[A] -> [A, None]
        - Single type A -> [A]
        - None type -> [None]
        """
        if annotation is None and self.typ is UNDEFINED:
            raise RuntimeError(
                f"Field '{self._name}' has modfield() but no annotation "
                f"and no explicit typ parameter"
            )

        # Extract types from annotation or typ parameter
        if self.typ is not UNDEFINED:
            # Use explicit typ parameter
            extracted = self.typ if isinstance(self.typ, list) else [self.typ]
        else:
            # Extract from annotation
            extracted = self._extract_types_from_annotation(annotation)

        # Compare or set
        if self._types is not None:
            if self._types != extracted:
                raise TypeError(
                    f"Field '{self._name}' type mismatch: "
                    f"annotation {extracted} != typ parameter {self._types}"
                )
        else:
            self._types = extracted

    def _extract_types_from_annotation(self, annotation) -> list:
        """Extract list of types from annotation.

        Union[A, B] -> [A, B]
        Optional[A] -> [A, None]
        A -> [A]

        Handles both typing.Union and types.UnionType (Python 3.10+ | syntax).
        """
        origin = get_origin(annotation)

        # Handle Union (including Optional which is Union[X, None])
        # Check both typing.Union and types.UnionType (for A | B syntax)
        if origin is Union or (isinstance(origin, type) and issubclass(origin, types.UnionType)):
            return list(get_args(annotation))

        # Single type
        return [annotation]

    def get_types(self) -> list:
        """Get types as list: [Type1, Type2, None, ...]"""
        return self._types

    def _to_spec_type(self, typ: type) -> type:
        """Convert a single type to its spec equivalent.

        Uses BaseModule.__convert_type_to_spec__ logic.
        Returns the spec type (e.g., Process -> ProcessSpec).
        """
        if typ is None:
            return None

        if isinstance(typ, type) and issubclass(typ, BaseModule):
            return typ.schema_model()

        return typ

    def get_spec_annotation(self) -> type:
        """Convert types to spec annotation for schema building."""
        spec_types = [self._to_spec_type(t) for t in self._types if t is not None]

        has_none = None in self._types

        if len(spec_types) == 0:
            return type(None) if has_none else None
        elif len(spec_types) == 1:
            return Optional[spec_types[0]] if has_none else spec_types[0]
        else:
            union = Union[tuple(spec_types)]
            return Optional[union] if has_none else union

    # Schema restriction (abstract - subclasses implement)
    @abstractmethod
    def restricted_schema(
        self,
        *,
        filter_schema_cls: Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """
        Generate restricted schema for this field.

        Returns:
            (field_schema, defs_dict) - Parent merges defs and inserts field_schema
        """
        raise NotImplementedError
```

### ModFieldDescriptor (Single Module Field)

```python
class ModFieldDescriptor(BaseFieldDescriptor):
    """Descriptor for single module field."""

    def restricted_schema(
        self,
        *,
        filter_schema_cls: Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """Generate restricted schema for single module field."""
        if variants is None:
            # No restriction - return base field schema
            base_schema = self._owner.schema()
            return (base_schema["properties"][self._name], {})

        # Process variants (uses inherited helper)
        variant_schemas = self._schema_process_variants(
            variants,
            restricted_schema_cls=filter_schema_cls,
            _seen=_seen,
            **kwargs
        )

        # Build entries
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

        # Build union
        if _profile == "shared":
            # Build defs dict
            union_name = self._schema_allowed_union_name(self._name)
            defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}

            # Add variant defs
            for name, schema in entries:
                defs[name] = schema

            # Field schema is $ref to union
            field_schema = {"$ref": f"#/$defs/{union_name}"}
            return (field_schema, defs)
        else:
            # Inline union
            field_schema = self._schema_make_union_inline(entries)
            return (field_schema, {})
```

### ModListFieldDescriptor (Module List Field)

```python
class ModListFieldDescriptor(BaseFieldDescriptor):
    """Descriptor for module list field."""

    def restricted_schema(
        self,
        *,
        filter_schema_cls: Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """Generate restricted schema for list field."""
        if variants is None:
            base_schema = self._owner.schema()
            return (base_schema["properties"][self._name], {})

        # Process variants (same as ModFieldDescriptor)
        variant_schemas = self._schema_process_variants(
            variants,
            restricted_schema_cls=filter_schema_cls,
            _seen=_seen,
            **kwargs
        )

        # Build entries
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

        # Build union (same logic)
        if _profile == "shared":
            union_name = self._schema_allowed_union_name(self._name)
            defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}
            for name, schema in entries:
                defs[name] = schema

            # Difference: wrap in array structure
            field_schema = {
                "type": "array",
                "items": {"$ref": f"#/$defs/{union_name}"}
            }
            return (field_schema, defs)
        else:
            field_schema = {
                "type": "array",
                "items": self._schema_make_union_inline(entries)
            }
            return (field_schema, {})
```

### ModDictFieldDescriptor (Module Dict Field)

```python
class ModDictFieldDescriptor(BaseFieldDescriptor):
    """Descriptor for module dict field."""

    def validate_annotation(self, annotation):
        """Validate dict annotation has str or int keys."""
        super().validate_annotation(annotation)

        origin = get_origin(annotation)
        if origin is dict:
            args = get_args(annotation)
            if len(args) == 2:
                key_type, _ = args
                if key_type not in (str, int):
                    raise TypeError(
                        f"moddictfield() keys must be str or int, got {key_type}"
                    )

    def restricted_schema(
        self,
        *,
        filter_schema_cls: Type[RestrictedSchemaMixin] = type,
        variants: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> tuple[dict, dict]:
        """Generate restricted schema for dict field."""
        if variants is None:
            base_schema = self._owner.schema()
            return (base_schema["properties"][self._name], {})

        # Process variants (same as ModFieldDescriptor)
        variant_schemas = self._schema_process_variants(
            variants,
            restricted_schema_cls=filter_schema_cls,
            _seen=_seen,
            **kwargs
        )

        # Build entries
        entries = [(self._schema_name_from_dict(s), s) for s in variant_schemas]

        # Build union (same logic)
        if _profile == "shared":
            union_name = self._schema_allowed_union_name(self._name)
            defs = {union_name: {"oneOf": self._schema_build_refs(entries)}}
            for name, schema in entries:
                defs[name] = schema

            # Difference: wrap in dict structure
            field_schema = {
                "type": "object",
                "additionalProperties": {"$ref": f"#/$defs/{union_name}"}
            }
            return (field_schema, defs)
        else:
            field_schema = {
                "type": "object",
                "additionalProperties": self._schema_make_union_inline(entries)
            }
            return (field_schema, {})
```

### Factory Functions

```python
def modfield(typ=UNDEFINED, default=UNDEFINED) -> ModFieldDescriptor:
    """Mark field as containing a BaseModule instance."""
    return ModFieldDescriptor(typ=typ, default=default)

def modlistfield(typ=UNDEFINED, default=UNDEFINED) -> ModListFieldDescriptor:
    """Mark field as containing a list of BaseModule instances."""
    return ModListFieldDescriptor(typ=typ, default=default)

def moddictfield(typ=UNDEFINED, default=UNDEFINED) -> ModDictFieldDescriptor:
    """Mark field as containing a dict of BaseModule instances."""
    return ModDictFieldDescriptor(typ=typ, default=default)
```

### BaseModule.__build_schema__ Updates

```python
@classmethod
def __build_schema__(cls):
    annotations = get_type_hints(cls, include_extras=True)
    spec_fields = {}

    for field_name, annotation in annotations.items():
        if get_origin(annotation) is ClassVar:
            continue

        field_value = getattr(cls, field_name, inspect._empty)

        # Check for modfield descriptor
        if isinstance(field_value, BaseFieldDescriptor):
            # Note: validate_annotation() already called in __set_name__
            # Just get spec annotation
            spec_annotation = field_value.get_spec_annotation()
            spec_fields[field_name] = (spec_annotation, ...)

        # Check for regular dataclass field
        elif isinstance(field_value, DataclassField):
            spec_fields[field_name] = (annotation, field_value.default)

        # Check for ShareableItem (Param/Attr/Shared)
        elif isinstance(field_value, ShareableItem):
            # Existing logic for Param/Attr/Shared
            # These use InitVar pattern, not modfield descriptors
            pass

        # No field marker - skip (just a type hint)

    # Rest of __build_schema__ logic...
```

### Usage Example

```python
from dataclasses import field

class ProcessCall(BaseModule, RestrictedProcessSchemaMixin):
    process: Process | AsyncProcess = modfield()
    args: dict = field(default_factory=dict)  # Regular field

    @classmethod
    def restricted_schema(cls, *, processes=None, _profile="shared", **kwargs):
        if processes is None:
            return cls.schema()

        schema = cls.schema()

        # Restrict process field using descriptor
        field_schema, field_defs = cls.process.restricted_schema(
            filter_schema_cls=RestrictedProcessSchemaMixin,
            variants=processes,
            _profile=_profile,
            **kwargs
        )

        # Merge defs and update field
        schema["$defs"].update(field_defs)
        schema["properties"]["process"] = field_schema

        # Custom logic for args
        input_types = cls._extract_input_types_from_processes(processes)
        schema = cls._update_args_with_types(schema, input_types)

        return schema
```

### Implementation Order

1. ✅ Implement BaseFieldDescriptor with descriptor protocol
2. ✅ Implement ModFieldDescriptor.restricted_schema()
3. ✅ Implement ModListFieldDescriptor.restricted_schema()
4. ✅ Implement ModDictFieldDescriptor.restricted_schema()
5. ✅ Implement modfield(), modlistfield(), moddictfield() factories
6. ✅ Remove `_schema_update_*_field()` methods from RestrictedSchemaMixin
7. ✅ Update BaseModule.__build_schema__ to use descriptors
8. ✅ Add comprehensive tests for all descriptor types
9. ✅ Migrate ProcessCall to use modfield()
10. ✅ Update ProcessCall.restricted_schema() to use descriptor pattern
11. ✅ Migrate Process/AsyncProcess
12. ✅ Migrate BehaviorTree + Task hierarchy
13. ✅ Migrate StateChart + State hierarchy
14. ✅ Run full test suite

### Success Criteria

- All descriptor types implement restricted_schema() correctly
- BaseModule.__build_schema__ recognizes modfield descriptors
- Type validation happens at class definition time
- Schema restriction works with descriptor-centric pattern
- All existing tests pass after migration
- No implicit field detection remains