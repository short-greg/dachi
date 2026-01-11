# Test Plan: Generics and Serialization Coverage

## Overview
This plan identifies missing test coverage for generic type parameters and model serialization/deserialization (round-trip) testing in the `proc` and `act` modules.

## Testing Principles Applied

1. **Test Public API only** - Focus on public methods and properties
2. **One Result Per Test** - Each test verifies one specific outcome
3. **Risk-Based Coverage** - Prioritize high-risk areas (data integrity, type safety, serialization)
4. **Clear Test Names** - `test_<method>_<expected_result>_<under_condition>`

---

## Module: `proc` (Process Module)

### Current State
- ✅ Basic Process functionality well-tested
- ✅ ProcessCall creation tested
- ❌ **NO generic type parameter tests**
- ❌ **NO serialization round-trip tests** (using `to_spec()`/`from_spec()`)
- ❌ **NO negative validation tests for generics**

### Note on Serialization API
**Use the wrapper methods instead of Pydantic directly:**
- `instance.to_spec()` instead of `instance.model_dump()`
- `ClassName.from_spec(spec)` instead of `ClassName.model_validate(spec)`
- `ClassName.to_schema()` instead of `ClassName.model_json_schema()`

These wrappers provide a more intuitive API and are the preferred methods for this codebase.

### Classes Using Generics (HIGH PRIORITY)

#### 1. `BaseProcessCall[ARGS]` and `ProcessCall[PROCESS, ARGS]`
**Location**: `dachi/proc/_process.py:223, 239`

**Test File**: `tests/proc/test_process_call.py`

**Missing Tests**:
```python
class TestProcessCallGenericTypeParameters:
    # Positive tests
    - test_process_call_preserves_generic_arg_model_type
    - test_process_call_works_with_custom_arg_model
    - test_process_call_with_different_process_types

class TestProcessCallSerialization:
    # Round-trip tests
    - test_to_spec_preserves_process_and_args
    - test_from_spec_reconstructs_process_call
    - test_spec_roundtrip_with_refs
    - test_spec_roundtrip_with_complex_args
    - test_serialization_preserves_arg_types

    # Negative tests
    - test_from_spec_rejects_invalid_structure
    - test_from_spec_rejects_wrong_arg_types
```

**Risk Level**: 🔴 HIGH - ProcessCall is core to graph execution

---

#### 2. `StreamSequence[PROCESS, STREAM]` and `AsyncStreamSequence[PA, ASYNC_STREAM]`
**Location**: `dachi/proc/_process.py:1164, 1194`

**Test File**: `tests/proc/test_process.py`

**Missing Tests**:
```python
class TestStreamSequenceGenericTypes:
    # Positive tests
    - test_stream_sequence_preserves_process_type
    - test_stream_sequence_preserves_stream_type
    - test_async_stream_sequence_with_custom_process

class TestStreamSequenceSerialization:
    # Round-trip tests
    - test_to_spec_preserves_sequence_structure
    - test_from_spec_reconstructs_stream_sequence
    - test_spec_roundtrip_preserves_process_chain
```

**Risk Level**: 🟡 MEDIUM - Used for composition, less common

---

#### 3. `LLMOptim[L, C]` and `Critic[C, P]`
**Location**: `dachi/proc/_optim.py:33, 129`

**Test File**: `tests/proc/test_optim.py`

**Missing Tests**:
```python
class TestLLMOptimGenericTypes:
    # Positive tests
    - test_llm_optim_with_different_llm_types
    - test_llm_optim_with_different_critique_types

class TestLLMOptimSerialization:
    # Round-trip tests
    - test_to_spec_preserves_llm_and_critique
    - test_from_spec_reconstructs_optim

class TestCriticGenericTypes:
    # Positive tests
    - test_critic_with_custom_critique_type
    - test_critic_with_custom_process_type
```

**Risk Level**: 🟡 MEDIUM - Optimization is important but isolated

---

#### 4. `DataFlow[PROCESS_CALL]`
**Location**: `dachi/proc/_graph.py:402`

**Test File**: `tests/proc/test_graph.py`

**Missing Tests**:
```python
class TestDataFlowGenericTypes:
    # Positive tests
    - test_dataflow_preserves_process_call_type
    - test_dataflow_with_different_process_call_types

class TestDataFlowSerialization:
    # Round-trip tests (CRITICAL for graph persistence)
    - test_to_spec_preserves_entire_graph_structure
    - test_from_spec_reconstructs_graph_with_nodes
    - test_spec_roundtrip_with_complex_graph
    - test_spec_roundtrip_preserves_node_dependencies
    - test_spec_roundtrip_preserves_execution_order

    # Negative tests
    - test_from_spec_rejects_circular_dependencies
    - test_from_spec_rejects_invalid_graph_structure
```

**Risk Level**: 🔴 HIGH - Graph execution is core functionality, data integrity critical

---

#### 5. Argument Model Helpers: `KWOnly[V]`, `PosArgs[V]`, `KWArgs[V]`
**Location**: `dachi/proc/_arg_model.py:23, 28, 33`

**Test File**: `tests/proc/test_process_func_arg.py`

**Missing Tests**:
```python
class TestArgModelGenericTypes:
    # Positive tests
    - test_kwonly_preserves_value_type
    - test_posargs_preserves_value_type
    - test_kwargs_preserves_value_type
    - test_arg_models_with_complex_types

class TestArgModelSerialization:
    # Round-trip tests
    - test_kwonly_spec_roundtrip_preserves_type
    - test_posargs_spec_roundtrip_preserves_type
    - test_kwargs_spec_roundtrip_preserves_type
```

**Risk Level**: 🟡 MEDIUM - Helper classes, well-isolated

---

### Proc Module Test Summary

**Total New Test Classes**: 13
**Estimated New Tests**: ~40-50 tests
**Priority Order**:
1. 🔴 DataFlow serialization (graph persistence)
2. 🔴 ProcessCall serialization (execution integrity)
3. 🟡 StreamSequence generics
4. 🟡 LLMOptim/Critic generics
5. 🟡 Argument model helpers

---

## Module: `act` (Behavior Tree / StateChart Module)

### Current State
- ✅ StateChart and Region functionality well-tested (590 tests passing)
- ❌ **NO generic type parameter tests**
- ❌ **NO serialization round-trip tests**
- ❌ **NO tests for custom state subclasses**

### Classes Using Generics (HIGH PRIORITY)

#### 1. `StateChart[BASE_STATE]`
**Location**: `dachi/act/_chart/_chart.py:31`

**Test File**: `tests/act/chart/test_chart.py`

**Missing Tests**:
```python
class TestStateChartGenericTypeParameters:
    # Positive tests - using custom state types
    - test_statechart_accepts_custom_state_subclass
    - test_statechart_with_stream_state_type_parameter
    - test_statechart_with_custom_final_state_type
    - test_statechart_multiple_regions_same_state_type
    - test_statechart_preserves_state_subclass_fields

class TestStateChartSerialization:
    # Round-trip tests (CRITICAL for persistence/checkpointing)
    - test_to_spec_preserves_chart_structure
    - test_from_spec_reconstructs_statechart
    - test_spec_roundtrip_preserves_regions
    - test_spec_roundtrip_preserves_custom_state_fields
    - test_spec_roundtrip_with_multiple_custom_state_types
    - test_to_spec_excludes_private_runtime_attrs
    - test_from_spec_restores_default_runtime_values

    # Negative tests
    - test_from_spec_rejects_invalid_chart_structure
    - test_from_spec_rejects_missing_required_fields
    - test_from_spec_rejects_malformed_data
```

**Risk Level**: 🔴 HIGH - Core statechart execution, persistence critical

---

#### 2. `Region[BASE_STATE]`
**Location**: `dachi/act/_chart/_region.py:101`

**Test File**: `tests/act/chart/test_region.py`

**Missing Tests**:
```python
class TestRegionGenericTypeParameters:
    # Positive tests
    - test_region_accepts_custom_state_type
    - test_region_with_stream_state_type
    - test_region_with_mixed_state_subclasses
    - test_region_preserves_state_type_information
    - test_region_validates_state_type_at_add

class TestRegionSerialization:
    # Round-trip tests
    - test_to_spec_preserves_region_structure
    - test_from_spec_reconstructs_region_with_states
    - test_spec_roundtrip_preserves_rules
    - test_spec_roundtrip_preserves_all_state_types
    - test_spec_roundtrip_preserves_initial_state

    # Negative tests
    - test_from_spec_rejects_invalid_initial_state
    - test_from_spec_rejects_invalid_rule_structure
```

**Risk Level**: 🔴 HIGH - Region is fundamental building block

---

### Act Module Test Summary

**Total New Test Classes**: 4
**Estimated New Tests**: ~25-30 tests
**Priority Order**:
1. 🔴 StateChart serialization (persistence critical)
2. 🔴 Region serialization (building block)
3. 🔴 StateChart generic type parameters
4. 🔴 Region generic type parameters

---

## Implementation Strategy

### Phase 1: High-Priority Serialization Tests (Week 1)
Focus on data integrity and persistence:
1. `tests/proc/test_graph.py` - DataFlow serialization
2. `tests/proc/test_process_call.py` - ProcessCall serialization
3. `tests/act/chart/test_chart.py` - StateChart serialization
4. `tests/act/chart/test_region.py` - Region serialization

### Phase 2: Generic Type Parameter Tests (Week 2)
Ensure type safety:
1. `tests/act/chart/test_chart.py` - StateChart generics with custom states
2. `tests/act/chart/test_region.py` - Region generics
3. `tests/proc/test_process_call.py` - ProcessCall generics
4. `tests/proc/test_graph.py` - DataFlow generics

### Phase 3: Medium Priority (Week 3)
1. StreamSequence generics and serialization
2. LLMOptim/Critic generics
3. Argument model helpers

---

## Test Template Examples

### Serialization Round-Trip Template
```python
class Test<ClassName>Serialization:

    def test_to_spec_preserves_<aspect>(self):
        """to_spec() preserves <aspect> when serialized."""
        # Arrange: Create instance with specific configuration
        instance = ClassName(field1=value1, field2=value2)

        # Act: Serialize
        spec = instance.to_spec()

        # Assert: Key fields preserved
        assert spec["field1"] == value1
        assert spec["field2"] == value2

    def test_from_spec_reconstructs_<aspect>(self):
        """from_spec() reconstructs instance with <aspect>."""
        # Arrange: Create and serialize
        original = ClassName(field1=value1, field2=value2)
        spec = original.to_spec()

        # Act: Reconstruct
        restored = ClassName.from_spec(spec)

        # Assert: Matches original
        assert restored.field1 == original.field1
        assert restored.field2 == original.field2

    def test_spec_roundtrip_preserves_<aspect>(self):
        """Spec serialization round-trip preserves <aspect>."""
        # Arrange
        original = ClassName(field1=value1, field2=value2)

        # Act: Round-trip via spec
        spec = original.to_spec()
        restored = ClassName.from_spec(spec)

        # Assert: Data preserved
        assert restored.field1 == original.field1
        assert restored.field2 == original.field2
```

### Generic Type Parameter Template
```python
class Test<ClassName>GenericTypeParameters:

    def test_<class>_preserves_<type_param>_type(self):
        """<ClassName> preserves <TypeParam> type parameter."""
        # Arrange: Create with custom type
        instance = ClassName[CustomType](field=CustomType(data="test"))

        # Act: Access typed field
        result = instance.field

        # Assert: Type preserved
        assert isinstance(result, CustomType)
        assert result.data == "test"

    def test_<class>_works_with_different_<type_param>(self):
        """<ClassName> works with different <TypeParam> types."""
        # Arrange & Act: Create with different types
        instance1 = ClassName[TypeA](field=TypeA())
        instance2 = ClassName[TypeB](field=TypeB())

        # Assert: Both work correctly
        assert isinstance(instance1.field, TypeA)
        assert isinstance(instance2.field, TypeB)
```

### Negative Validation Template
```python
class Test<ClassName>NegativeValidation:

    def test_from_spec_rejects_<invalid_case>(self):
        """from_spec() rejects <invalid case> with ValidationError."""
        # Arrange: Invalid data
        invalid_spec = {"wrong": "structure"}

        # Act & Assert: Rejection
        with pytest.raises(ValidationError):
            ClassName.from_spec(invalid_spec)
```

---

## Success Criteria

### For Each Module:
- ✅ All generic type parameters have positive tests (custom types work)
- ✅ All key classes have `to_spec()` → `from_spec()` round-trip tests
- ✅ Tests verify spec serialization preserves structure
- ✅ Negative validation tests for malformed data
- ✅ Tests verify that private Runtime/Param/Attr are NOT serialized via to_spec()
- ✅ Tests verify that restored instances have correct default values
- ✅ All tests follow naming convention and test one result

### Overall:
- ✅ ~70-80 new tests added across proc and act modules
- ✅ Test coverage for generics >80%
- ✅ Test coverage for serialization >90%
- ✅ All tests pass
- ✅ No implementation details tested (only public API)

---

## Notes

- **State vs Spec**: Remember that `to_spec()` serializes the *spec* (configuration), not runtime state
- **Runtime Fields**: `Runtime`, `Param`, `Attr` fields (private attrs) should NOT appear in `to_spec()` output
- **Generic Preservation**: After round-trip via `from_spec()`, generic type information may be lost at runtime (Python limitation), but field types should be validated
- **Test Organization**: Add tests to existing test files, organized in new test classes following the naming pattern
- **Preferred API**: Always use `to_spec()`/`from_spec()`/`to_schema()` instead of the underlying Pydantic methods
