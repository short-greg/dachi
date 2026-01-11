# LLM Adapter Pattern Improvement Plan

## Summary
Objectives: Make LLM adapters more flexible and predictable by adding tools, base_args, and out parameters while consolidating the tool system and standardizing output handling.

## Requirements Analysis

### 1. Add tools parameter to adapters
- Users can provide a list of tools (functions wrapped in ToolDef) directly to the adapter
- The adapter will pass these tools to the LLM backend if it supports them
- Tools can be declared once at initialization, but users can still override or add tools during a call if needed
- This keeps tool integration explicit and avoids hidden behaviors

### 2. Add base_args parameter to adapters
- Users can pass a dictionary of default arguments (base_args) to the adapter when creating it
- These defaults are applied to every call made through the adapter
- Examples include things like temperature, max_tokens, or default tools
- Call-time arguments passed to forward, stream, etc. shallowly override base_args
- Example: base_args={"temperature": 0.7} but calling forward(..., temperature=0.9) will override it
- This balances convenience (set once) with flexibility (override when needed)

### 3. Add out parameter to adapters
The out parameter determines how the response is structured. It maps directly to Resp.out.

Supported options:
- **Primitive or built-in types**: e.g. out=bool → resp.out = bool(msg.text)
- **JSONObj wrapper**: 
  - Can be empty (JSONObj()) → use json_object
  - Or with a schema (JSONObj(schema=...)) → use json_schema
  - If given a Pydantic model, the schema is inferred automatically
- **Custom response processors (RespProc)**: The adapter will use the provided processor to handle the response
- **Tuples**: e.g. out=(RespProc1(), JSONObj(...))
  - Returns a flat tuple of outputs
  - Elements can mix processors and schema objects
  - System must check for conflicts (e.g. two outputs both trying to map the same key)
- **Dictionaries**: e.g. out={"a": JSONObj(...), "b": bool}
  - Returns a dict of values, keyed by the given names
  - This makes the adapter a single point of truth for how outputs are structured

### 4. Override behavior
- kwargs passed to forward, stream, astream, etc. override the base_args set at adapter creation
- The override is shallow (only the keys provided at call time are replaced)
- This ensures predictability and avoids deep merging surprises

## Current State Analysis

### Completed Tasks ✅
1. **Analyzed current OpenAI adapter implementation**: Found ChatCompletion, OpenAIChat, and OpenAIResp classes with separate converter system
2. **Identified ToolCall class differences**:
   - **ToolCall** (sync): has tool_id, option, inputs, result, option_text
   - **AsyncToolCall** (async): same fields but missing tool_id 
   - **LLMToolCall** (API response): has id, name, arguments, type, raw_arguments, is_complete
3. **Designed improved adapter interface**: Clarified requirements with user
4. **Removed LLMToolCall**: Not needed, creates unnecessary complexity
5. **Fixed AsyncToolCall**: Added missing tool_id field
6. **Added id field to ToolCall**: For cases where ToolDef is unknown, tools can still be located

### Tool System Status
- **ToolCall** and **AsyncToolCall** now have consistent interfaces
- Both have: tool_id, id (optional), option, inputs, result, option_text  
- LLMToolCall removed from codebase and imports
- Test file cleaned up

## Next Steps

### High Priority Tasks 🔴

#### 6. Create JSONObj wrapper and output conflict detection utilities
**Location**: `dachi/proc/_out.py`
**Implementation**:
```python
class JSONObj(pydantic.BaseModel):
    """Wrapper for JSON output configuration"""
    schema: typing.Dict | pydantic.BaseModel | None = None
    mode: typing.Literal["json_object", "json_schema"] | None = None
    
    def get_schema_dict(self) -> typing.Dict | None:
        # Convert Pydantic models to schema dict
    
    def to_response_format(self) -> typing.Dict:
        # Convert to OpenAI response_format parameter
```

**Conflict Detection Utilities**:
```python
def detect_output_conflicts(out_spec) -> typing.List[str]:
    """Detect conflicts in tuple/dict output specifications at creation time"""

def validate_output_runtime(resp: Resp, out_spec) -> None:
    """Validate output conflicts at runtime"""
```

#### 7. Implement improved adapter interface
**Location**: Create new base class, probably `dachi/adapt/_base.py`
**Interface**:
```python
class ImprovedLLMAdapter(BaseModule):
    tools: typing.List[ToolDef] | None = None
    base_args: typing.Dict[str, typing.Any] | None = None  
    out: typing.Union[
        type,  # Primitives like bool, str, int, float
        JSONObj,  # JSON with optional schema
        RespProc,  # Custom processor
        typing.Tuple,  # Multiple outputs
        typing.Dict[str, typing.Any]  # Named outputs
    ] | None = None
    
    def forward(self, inp, **kwargs) -> Resp:
        # Merge base_args with kwargs (shallow override)
        # Apply tools if provided
        # Process output according to out specification
```

#### 8. Update OpenAI adapters to use new interface
**Files**: `dachi/adapt/xopenai/_openai.py`, `dachi/proc/_ai.py`
**Tasks**:
- Replace ChatCompletion class with improved interface
- Update OpenAIChat and OpenAIResp to inherit from ImprovedLLMAdapter
- Remove old converter system in favor of unified out parameter
- Implement tools parameter integration
- Implement base_args merging logic

### Medium Priority Tasks 🟡

#### 9. Update tests to cover new adapter functionality
**Files**: `tests_adapt/adapt/test_xopenai.py`, `tests/proc/test_ai.py`
**Coverage needed**:
- Tools parameter functionality
- Base_args override behavior (creation time vs call time)
- All out parameter options (primitives, JSONObj, tuples, dicts, RespProc)
- Conflict detection (both creation time and runtime)
- Backward compatibility (if any maintained)

## Technical Design Decisions

### Questions Resolved ✅
1. **ToolCall Consolidation**: Keep separate ToolCall and AsyncToolCall classes, both now have consistent interfaces
2. **Backward Compatibility**: No need to maintain compatibility with existing ChatCompletion class
3. **Schema Auto-Detection**: No auto-inference - users must explicitly set schema dict or use Pydantic BaseModel
4. **Conflict Detection**: Must be done at both creation time and runtime, with general utilities provided

### Key Architecture Principles
1. **Single Point of Truth**: The adapter becomes the single place where output structure is defined
2. **Explicit over Implicit**: Tools and output formats must be explicitly specified
3. **Composability**: Support for mixing different output types (tuples, dicts)
4. **Predictable Overrides**: Shallow merging for arguments, clear precedence rules
5. **Conflict Prevention**: Early detection of conflicting output specifications

## Implementation Notes

### JSONObj Design
- Empty JSONObj() → json_object mode
- JSONObj(schema=dict) → json_schema mode with custom schema
- JSONObj(schema=PydanticModel) → json_schema mode with inferred schema
- Must provide to_response_format() method for OpenAI integration

### Output Conflict Detection
- **Creation Time**: Check for duplicate keys in dict outputs, incompatible tuple elements
- **Runtime**: Validate that processors don't try to write to same Resp fields
- **Error Messages**: Clear indication of what conflicts and how to resolve

### Base Args Merging
```python
final_args = {**self.base_args, **call_time_kwargs}
```
Simple shallow merge, call-time arguments win

## Files Modified So Far
- ✅ `/Users/shortg/Development/dachi/dachi/core/_tool.py` - Removed LLMToolCall, added id field to both ToolCall classes
- ✅ `/Users/shortg/Development/dachi/dachi/core/__init__.py` - Removed LLMToolCall export
- ✅ `/Users/shortg/Development/dachi/tests/proc/test_ai.py` - Removed LLMToolCall tests and import

## Current Session Progress ✅

### Completed Tasks
6. ✅ **Created JSONObj wrapper and output conflict detection utilities** in `dachi/proc/_out.py`
7. ✅ **Moved OpenAI adapters to dedicated module** - Created `dachi/proc/_openai.py` with:
   - Moved `OpenAIChat` and `OpenAIResp` from `_ai.py` to `_openai.py` 
   - Added optional `url` parameter for custom endpoints
   - Fixed streaming implementation with proper text accumulation
   - Added thinking/reasoning accumulation for Responses API
   - Updated imports in `__init__.py`
8. ✅ **Removed ChatCompletion class** from `dachi/adapt/xopenai/_openai.py` as no longer needed

### Files Modified This Session
- ✅ `/Users/shortg/Development/dachi/dachi/proc/_out.py` - Added JSONObj, detect_output_conflicts, validate_output_runtime
- ✅ `/Users/shortg/Development/dachi/dachi/proc/_openai.py` - New file with OpenAI adapter classes
- ✅ `/Users/shortg/Development/dachi/dachi/proc/__init__.py` - Updated imports
- ✅ `/Users/shortg/Development/dachi/dachi/proc/_ai.py` - Removed OpenAI classes
- ✅ `/Users/shortg/Development/dachi/dachi/adapt/xopenai/_openai.py` - Removed ChatCompletion class

## Next Session Priority
1. Add tools, base_args, and out parameter support to the OpenAI adapters
2. Update tests to cover new functionality 
3. Consider reusable components for future API adapters