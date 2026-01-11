# ModField Implementation - Session Summary

## Status: Ready for Implementation

This document provides a quick reference for the modfield implementation plan. Full details are in [restricted_schema_guide.md](./restricted_schema_guide.md) starting at line 3391.

## Quick Reference

### What We're Building
Explicit module field descriptors to replace the current implicit field detection system.

### Key Changes
1. **Explicit field markers**: `process: Process = modfield()` instead of just `process: Process`
2. **Descriptor-based**: Uses Python descriptor protocol for fields
3. **Schema restriction**: Descriptors handle their own restricted schema generation
4. **No backward compatibility**: Clean break - all modules must migrate

### Architecture

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

### Core API

```python
from dataclasses import field

class MyModule(BaseModule, RestrictedTaskSchemaMixin):
    # Module fields (use modfield)
    child: Task = modfield()
    children: list[Task] = modlistfield()
    states: dict[str, State] = moddictfield()

    # Regular fields (use dataclasses.field)
    count: int = field(default=0)
    data: dict = field(default_factory=dict)

    # Param/Attr/Shared (still use InitVar pattern in __post_init__)
    def __post_init__(self, value: int = 5):
        super().__post_init__()
        self.value = Param(value)
```

### Descriptor Return Type

All descriptors return `(field_schema, defs)` tuple:
```python
field_schema, field_defs = cls.child.restricted_schema(
    filter_schema_cls=RestrictedTaskSchemaMixin,
    variants=[Task1, Task2],
    _profile="shared"
)

schema["$defs"].update(field_defs)
schema["properties"]["child"] = field_schema
```

### Implementation Order

1. Implement BaseFieldDescriptor + 3 variants
2. Implement factory functions (modfield, modlistfield, moddictfield)
3. Remove old `_schema_update_*_field()` methods from RestrictedSchemaMixin
4. Update BaseModule.__build_schema__
5. Add comprehensive tests
6. Migrate modules:
   - ProcessCall
   - Process/AsyncProcess
   - BehaviorTree + Task hierarchy
   - StateChart + State hierarchy

### Helper Method Distribution

**Stay in RestrictedSchemaMixin**:
- `_schema_process_variants()` - Domain filtering
- `_schema_name_from_dict()`
- `_schema_require_defs_for_entries()`
- `_schema_build_refs()`
- `_schema_make_union_inline()`
- `_schema_allowed_union_name()`
- `_schema_ensure_shared_union()`
- `_schema_merge_defs()`
- `_schema_node_at()`
- `_schema_replace_at_path()`

**Remove from RestrictedSchemaMixin**:
- `_schema_update_list_field()` → ModListFieldDescriptor.restricted_schema()
- `_schema_update_dict_field()` → ModDictFieldDescriptor.restricted_schema()
- `_schema_update_single_field()` → ModFieldDescriptor.restricted_schema()

### Key Implementation Notes

1. **UNDEFINED sentinel**: Use `inspect._empty`
2. **Union detection**: Handle both `typing.Union` and `types.UnionType` (Python 3.10+)
3. **Type validation**: Happens in `__set_name__`, not `__build_schema__`
4. **Nullable fields**: None added to type list, wrapped in Optional
5. **Imports needed**:
   ```python
   from abc import ABC, abstractmethod
   from typing import Union, Optional, get_origin, get_args, Type
   from inspect import _empty as UNDEFINED
   import types
   ```

### Files to Modify

**Core**:
- `dachi/core/_base.py` - Add descriptors, update __build_schema__, remove old helpers from RestrictedSchemaMixin

**Migrations**:
- `dachi/proc/_graph.py` - ProcessCall
- `dachi/proc/_process.py` - Process/AsyncProcess
- `dachi/act/_task.py` - Task hierarchy
- `dachi/act/_behavior_tree.py` - BehaviorTree
- `dachi/act/state/_statechart.py` - StateChart hierarchy

**Tests**:
- `tests/core/test_base.py` - Add descriptor tests
- Update all existing tests after migration

### Success Criteria

- ✅ All descriptor types implement restricted_schema() correctly
- ✅ BaseModule.__build_schema__ recognizes modfield descriptors
- ✅ Type validation happens at class definition time
- ✅ Schema restriction works with descriptor-centric pattern
- ✅ All existing tests pass after migration
- ✅ No implicit field detection remains

## Plan Review Complete

The plan has been reviewed and confirmed ready for implementation. All issues identified during review have been addressed:

1. ✅ Added proper imports (ABC, abstractmethod, typing imports, types module)
2. ✅ Fixed Union type detection to handle both typing.Union and types.UnionType
3. ✅ Consistent method signatures across all descriptor types
4. ✅ Removed duplicate validate_annotation() call
5. ✅ Clarified ShareableItem (Param/Attr/Shared) use InitVar pattern
6. ✅ Documented nullable handling (None in type list)
7. ✅ Specified UNDEFINED as inspect._empty

**Status**: Ready to implement when session restarts.

## Next Session

Start with task 1: Implement BaseFieldDescriptor with descriptor protocol in `dachi/core/_base.py`.
