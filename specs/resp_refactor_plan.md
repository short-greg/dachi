# Response Processor Refactoring Plan

## Overview
The `dachi/proc/_resp.py` file contains response processors used to convert LLM outputs into structured data and populate the `.out` member of `Resp` objects. This file needs significant cleanup to remove deprecated classes and simplify the architecture.

## Current State
- Mixed architecture with both `RespProc` and `ToOut` base classes
- Many deprecated classes marked with TODO comments  
- Redundant functionality between different processor types
- Complex field management with `name` and `from_` parameters
- Parser classes with unclear relationships to output processors

## Target Architecture
- Single `ToOut` base class for all output processors
- Field management handled entirely in `_ai.py` (not in processors)
- Clean separation between user-facing `out` specifications and internal processors
- Simplified streaming support

## User Interface (handled in _ai.py)
Users can specify `out` parameter as:
- `out=bool` → Creates `ToPrim('bool')` processor
- `out=dict(x=bool, y=SomeToOut())` → Creates appropriate processors for each field
- `out=(bool, int)` → Creates tuple of processors, returns tuple
- `out=SomeToOut()` → Uses the ToOut subclass directly

## Refactoring Plan

### Phase 1: Remove Deprecated Classes (High Priority)

#### RespProc Classes to Remove:
- `RespProc` (base class)
- `TextConv` - redundant, adapters handle automatically
- `StructConv` - redundant, adapters handle automatically  
- `StructStreamConv` - redundant, adapters handle automatically
- `ToolConv` - redundant, adapters handle automatically
- `ParsedConv` - redundant, use StructConv instead
- `FromResp` - not necessary anymore

#### Deprecated ToOut Classes to Remove:
- `StrOut` - not necessary, assume string by default
- `PydanticOut` - not necessary anymore  
- `JSONOut` - not necessary, use StructConv
- `ListOut` - remove, use other mechanisms
- `ParseOut` - remove, since parsers being simplified

#### Utility Functions to Remove:
- `conv_to_out()` - logic moves to `_ai.py`
- `detect_output_conflicts()` and `validate_output_runtime()` - logic moves to `_ai.py`

### Phase 2: Clean Up Remaining Classes (Medium Priority)

#### ToOut Classes to Keep & Refactor:
- `ToOut` (base class) - remove `name`, `from_` fields
- `PrimOut` → rename to `ToPrim` - for primitive type conversion (bool, int, float, str)
- `KVOut` - for key-value parsing with separators
- `IndexOut` - for indexed list parsing  
- `CSVOut` - for CSV data parsing
- `TupleOut` - for tuple outputs
- `ToolExecConv` - for tool execution (if still needed)

#### Parser Classes to Review:
- `LineParser` - keep, used by other processors
- `CharDelimParser` - review relationship to `KVOut`, possibly remove
- `CSVRowParser` - ensure integration with `CSVOut`
- `CSVCellParser` - remove (marked for removal)

### Phase 3: Simplify Architecture (Medium Priority)

#### Remove Field Management:
- Remove `name` and `from_` parameters from all classes
- Remove field mapping logic in `forward()` methods
- Simplify `delta()` method signatures where possible

#### Streamline Processing:
- Keep essential streaming support (`is_streamed`, `is_last`)
- Remove redundant `post()` methods where not needed
- Simplify `forward()` method to focus on core processing

## Implementation Steps

1. **Remove deprecated RespProc classes and their imports**
2. **Remove deprecated ToOut classes**
3. **Remove utility functions**
4. **Rename PrimOut to ToPrim**
5. **Remove name/from_ fields from remaining classes**
6. **Simplify delta/forward methods**
7. **Clean up Parser classes**
8. **Update tests and imports**

## Expected Benefits

- Cleaner, more maintainable codebase
- Clear separation of concerns
- Simplified user interface
- Better integration with adapter system
- Reduced complexity and redundancy

## Files Affected

- `dachi/proc/_resp.py` - main refactoring target
- `dachi/proc/_ai.py` - will handle out parameter conversion
- Test files in `tests/proc/` - will need updates
- Any imports of deprecated classes throughout codebase

## Testing Strategy

- Maintain existing functionality for kept classes
- Update tests to reflect new simplified interface
- Ensure streaming behavior still works correctly
- Verify integration with adapter system