# Tool Adaptation System

## Overview

The tool adaptation system in Dachi provides a unified interface for defining, registering, and executing tools that can be called by LLMs. The system uses a registry-based architecture where functions are stored separately from their metadata, enabling efficient lookup and execution while maintaining clean separation of concerns.

## Architecture Components

### Core Components (`dachi/core/_tool.py`)

#### Private Tool Registry
- **Private registry**: `_tool_registry: Dict[str, Callable] = {}` - Private registry mapping qualified names to functions
- **Access functions**: 
  - `get_tool_function(name: str) -> Callable` - Retrieve function by qualified name
  - `list_tool_functions() -> Dict[str, Callable]` - Get all registered functions

#### Tool Definition Hierarchy
- **BaseTool**: Abstract base class defining the tool interface
  - `name`: Tool identifier (uses function's `__qualname__`)
  - `description`: Generated from docstring or auto-generated
  - `input_model`: Auto-generated Pydantic model from function signature
  - `return_type`: Optional return type annotation
  - `is_async`: Abstract property indicating sync/async execution
  
- **Tool**: Synchronous tool implementation
  - Implements `__call__()` to execute the registered function
  - Uses registry lookup: `get_tool_function(self.name)` 
  
- **AsyncTool**: Asynchronous tool implementation
  - Implements `async __call__()` to execute async functions
  - Same registry lookup pattern as Tool

#### Tool Registration System

**Function Registration**: `register_tool(func: Callable) -> BaseTool`
- Stores function in private registry using `func.__qualname__` as key
- Generates Pydantic input model from function signature
- Returns appropriate Tool or AsyncTool instance based on `is_async_function()`
- Validates function signatures (no *args/**kwargs allowed)

**Decorator Interface**: `@tool`
```python
@tool
def calculate_sum(x: int, y: int) -> int:
    """Calculate the sum of two integers."""
    return x + y
```
- Replaces decorated function with Tool/AsyncTool instance
- Automatically calls `register_tool()` internally
- Function becomes callable Tool object that manages execution

#### Tool Execution Model

**Separation of Storage and Metadata**:
- **Functions**: Stored in private `_tool_registry` by qualified name
- **Metadata**: Stored on Tool/AsyncTool instances (name, description, input_model, etc.)
- **Execution**: Tool instances look up and call functions via registry

**ToolUse**: Represents a tool call instance with inputs and execution
- `tool_id`: Call identifier from LLM
- `option`: Tool definition (BaseTool instance)
- `inputs`: Pydantic model instance with validated parameters
- `result`: Execution result (set after calling)
- `executed`: Boolean flag tracking execution state

**Execution Methods**:
- `__call__()`: Execute tool (delegates to `forward()`)
- `forward()`: Sync execution - calls `get_tool_function()` and executes
- `aforward()`: Async execution - supports both sync and async functions
- `to_tool_call(*args, tool_id: str, **kwargs) -> ToolUse`: Create ToolUse instance

### Streaming Buffer System

#### ToolChunk
Represents a single partial update for one tool call during streaming:

**Routing Fields** (stable across deltas):
- `id`: Optional provider-specific tool call ID  
- `turn_index`: Assistant message/content-block index
- `call_index`: Per-message tool call index

**Content Fields**:
- `name`: Tool/function name
- `args_text_delta`: JSON text fragment for incremental parsing
- `args_kv_patch`: Key-value updates for direct argument building

**Lifecycle Fields**:
- `done`: Completion signal (set True once per tool call)
- `error`/`error_type`: Optional error information

**Extensibility**:
- `metadata`: Dict for provider-specific data preservation

#### ToolBuffer
Accumulates tool call chunks and manages their lifecycle:

**Key Components**:
- `_acc`: Dict mapping routing keys to partial tool call data
- `_tool_map`: Dict mapping tool names to Tool instances
- `_calls`: List of completed ToolUse instances
- `_chunks`: Historical chunk storage (optional)

**Routing Strategy**:
- **Primary**: Use provider ID if available (`chunk.id`)
- **Fallback**: Use index pair (`turn_index`, `call_index`)  
- **Key generation**: `_make_key(chunk)` creates stable routing keys

**Accumulation Process**:
1. **Route**: Determine which accumulator to update based on routing key
2. **Accumulate**: Add text fragments or merge key-value patches
3. **Detect**: Check for completion (`done=True` + required fields present)
4. **Finalize**: Create ToolUse instance and clear accumulator

**JSON Assembly Strategies**:
- **Text fragments**: Concatenate `args_text_delta` strings, parse as JSON
- **Key-value patches**: Merge `args_kv_patch` dicts directly
- **Fallback**: JSON repair for malformed streaming data

**Completion Logic**:
- Requires: `acc["done"]` and `acc["name"]` 
- ID optional: Supports both ID-based and index-based routing
- Auto-generates tool_id for index-based calls: `f"tool_{turn}_{call}"`

## LLM Provider Integration

### Request Preparation (Tools → Provider Format)
Tools are converted to provider-specific formats:

**OpenAI Chat Completions**:
```python
{
    "type": "function", 
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_model.model_json_schema()
    }
}
```

### Response Processing (Provider Format → ToolUse)

**Non-Streaming**: Direct parsing of complete tool calls
**Streaming**: Multi-step process via ToolBuffer

#### Streaming Response Flow

**1. Provider-Specific Parsing**:
```python
# OpenAI delta example
{
    "choices": [{
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_abc123",
                "function": {
                    "name": "calculate_sum",
                    "arguments": '{"x": 5'
                }
            }]
        }
    }]
}

# Map to ToolChunk
ToolChunk(
    id="call_abc123",
    call_index=0,
    name="calculate_sum", 
    args_text_delta='{"x": 5',
    done=False
)
```

**2. Buffer Accumulation**:
- ToolBuffer routes chunk to appropriate accumulator
- Concatenates JSON fragments or merges KV patches
- Tracks completion via `done` flag

**3. Tool Call Completion**:
- Parse accumulated arguments (with JSON repair if needed)
- Look up tool definition in `_tool_map` 
- Create and validate ToolUse instance
- Clear accumulator for completed call

**4. Execution** (optional):
- Call `tool_use.forward()` or `tool_use.aforward()`
- Store result in `tool_use.result`

## Provider Mapping Patterns

### OpenAI Chat Completions
```python
# Streaming tool call format
{
    "choices": [{
        "delta": {
            "tool_calls": [{
                "index": 0,               # → call_index
                "id": "call_xyz",         # → id  
                "function": {
                    "name": "tool_name",  # → name
                    "arguments": "json"   # → args_text_delta
                }
            }]
        },
        "finish_reason": "tool_calls"     # → done=True when finished
    }]
}
```

### Anthropic Claude
```python  
# Content block streaming
{
    "type": "content_block_delta",
    "index": 0,                           # → call_index
    "delta": {
        "type": "input_json_delta",
        "partial_json": '"x": 5, "y": 3'  # → args_text_delta
    }
}
# Completion detected by content_block_stop event
```

### Google Gemini
```python
# Function call format (usually complete)
{
    "candidates": [{
        "content": {
            "parts": [{
                "functionCall": {
                    "name": "tool_name",     # → name
                    "args": {"x": 5, "y": 3} # → args_kv_patch (complete)
                }
            }]
        }
    }]
}
```

### Cohere
```python
# Multi-event sequence
{"event": "tool-call-start", "tool_call_id": "call_123"}    # → id, start accumulator
{"event": "tool-call-delta", "delta": {"name": "tool"}}     # → name
{"event": "tool-call-delta", "delta": {"parameters": "{}"}} # → args_text_delta  
{"event": "tool-call-end", "tool_call_id": "call_123"}      # → done=True
```

## Parallel Tool Call Support

### Multiple Simultaneous Calls
The system fully supports parallel tool calls from LLMs:

**Key-based Routing**: Each tool call gets a unique routing key:
- `("call_1", None, None)` - ID-based routing
- `(None, 0, 0)` - Index-based routing  
- `(None, 0, 1)` - Second parallel call

**Interleaved Processing**: ToolBuffer handles chunks arriving in any order:
```python
# Chunks can arrive interleaved from multiple tool calls
ToolChunk(id="call_1", name="tool_a")     # Start call 1
ToolChunk(id="call_2", name="tool_b")     # Start call 2  
ToolChunk(id="call_1", args_text_delta="...") # Continue call 1
ToolChunk(id="call_2", args_text_delta="...") # Continue call 2
ToolChunk(id="call_1", done=True)         # Complete call 1
ToolChunk(id="call_2", done=True)         # Complete call 2
```

**Completion Independence**: Each tool call completes independently and is processed immediately.

## Error Handling

### ToolBuffer Error Processing
- **JSON Parse Errors**: Automatic repair with fallback to empty dict
- **Unknown Tools**: KeyError with descriptive message
- **Validation Errors**: Pydantic validation with field-level details
- **Provider Errors**: Optional error fields in ToolChunk with error propagation

### Tool Execution Errors
- **Sync Tools**: Exceptions bubble up normally
- **Async Tools**: Await exceptions in `aforward()`
- **Type Validation**: Pydantic ensures input model compliance
- **Missing Functions**: Registry lookup failures raise KeyError

## Usage Patterns

### Basic Tool Definition
```python
@tool
def calculate_sum(x: int, y: int) -> int:
    """Calculate the sum of two integers."""
    return x + y

# Function replaced with Tool instance
# Registered in private _tool_registry as:
# "_tool_registry['__main__.calculate_sum'] = <original_function>"
```

### Manual Tool Registration
```python
def my_function(value: str) -> str:
    return value.upper()

tool_def = register_tool(my_function)
# Returns Tool instance with auto-generated input model
```

### Tool Usage in Processing
```python
# Tool buffer for streaming responses
buffer = ToolBuffer(tools=[calculate_sum_tool, other_tool])

# Process streaming chunks
for chunk in streaming_chunks:
    completed = buffer.append(chunk)
    if completed:
        # Tool call completed, execute if desired
        result = completed.forward()
```

### Manual Tool Execution
```python
# Create tool use instance
tool_use = calculate_sum_tool.to_tool_call(x=5, y=3, tool_id="call_123")

# Execute synchronously  
result = tool_use()  # or tool_use.forward()

# Execute asynchronously
result = await tool_use.aforward()
```

## Design Principles

### 1. Registry-Based Architecture
- **Separation**: Functions stored separately from metadata
- **Efficiency**: O(1) lookup by qualified name
- **Flexibility**: Metadata can be updated without affecting function storage

### 2. Provider Agnostic Design  
- **Unified Interface**: Same tool definition works across all providers
- **Flexible Mapping**: ToolChunk/ToolBuffer handles diverse provider formats
- **Extensible**: Provider-specific metadata preserved via extensibility fields

### 3. Streaming-First Architecture
- **Incremental Processing**: Handle partial data as it arrives
- **Parallel Support**: Multiple concurrent tool calls
- **Robust Parsing**: JSON repair and error recovery

### 4. Type Safety & Validation
- **Pydantic Integration**: Auto-generated input models from function signatures  
- **Runtime Validation**: Input validation before function execution
- **Type Hints**: Full typing support for IDE integration

### 5. Async/Sync Transparency
- **Unified Interface**: Same API for sync and async tools
- **Detection**: Automatic sync/async detection via `is_async_function()`
- **Execution**: Both execution paths supported in ToolUse

## Integration Points

### Core Module System
- **Registry Integration**: Tools discoverable via global registry
- **Module Compatibility**: Tools can be wrapped as BaseModule instances  
- **State Management**: Tool definitions participate in spec/state system

### Processing Pipeline
- **Response Processing**: Tools integrated into resp processing chain
- **Message System**: Tool results become follow-up messages
- **Streaming Support**: Real-time tool call detection and execution

### LLM Adapters
- **Request Preparation**: Tools → provider-specific format conversion
- **Response Parsing**: Provider format → ToolChunk stream  
- **Execution Integration**: Optional automatic tool execution

## Implementation Status

### Completed Components
- ✅ **Private Registry System**: Qualified name-based function storage
- ✅ **Tool/AsyncTool Classes**: Registry-based execution with metadata
- ✅ **Auto-Registration**: @tool decorator and register_tool() function
- ✅ **ToolBuffer Implementation**: Streaming accumulation with parallel support
- ✅ **Provider Mapping**: Support for OpenAI, Claude, Gemini, Cohere patterns
- ✅ **Error Handling**: JSON repair, validation, and error propagation
- ✅ **Test Coverage**: Comprehensive test suite including edge cases

### Key Features
- **Qualified Names**: Uses `__qualname__` for unique tool identification
- **Index-based Routing**: Supports both ID and index-based tool call routing
- **JSON Repair**: Automatic repair of malformed streaming JSON
- **Parallel Processing**: Full support for concurrent tool calls
- **Type Safety**: Auto-generated Pydantic models with validation
- **Extensibility**: Provider metadata preservation and custom fields

The tool system is production-ready and provides a robust foundation for LLM tool calling across different providers and execution contexts.