# OpenAI Message & Tool Call Handling - Complete Redesign Plan

**Status**: Planning Phase
**Created**: 2025-11-30
**Author**: Claude (with human guidance)

---

## Meta-Plan: High-Level Structure

This document contains both the high-level plan and detailed subplans as they are completed.

### Overview

Redesign the OpenAI message and tool call handling system to:
- Complete all incomplete/stubbed implementations
- Simplify and modularize the code
- Support both Chat Completions and Responses APIs correctly
- Handle streaming properly with tool call reconstruction
- Maintain clean architecture with clear responsibilities

### Phase 1: Analysis & Requirements

#### Subplan 1.1: Current State Audit
**Status**: ✅ COMPLETED
**Purpose**: Understand what exists, what's broken, what's complete

**Activities**:
- Map complete data flow: OpenAI API → adapters → Msg/Resp/DeltaResp
- Document current field usage in Msg, Resp, DeltaResp, ToolUse
- Identify all incomplete code (stubs, comments, pass statements)
- Test current functionality to see what works vs breaks

**Acceptance Criteria**:
- [x] Complete inventory of all message fields and their current usage
- [x] List of all incomplete/broken code with line numbers
- [x] Clear understanding of what works today (if anything)
- [x] Zero ambiguity about current state

**Findings**:

##### Message Data Flow (Current)

**Non-streaming**:
```
OpenAI API Response
  ↓
OpenAIChat.from_result() OR OpenAIResp.from_result()
  ↓
Resp object (with text, tool_use, usage, etc.)
  ↓
Returned to user
```

**Streaming**:
```
OpenAI API Stream Chunks
  ↓
OpenAIChat.from_streamed_result() OR OpenAIResp.from_streamed_result()
  ↓  (with prev_resp for accumulation)
Tuple[Resp, DeltaResp]
  ↓
Yielded to user
```

##### Field Inventory

**Msg** (dachi/core/_msg.py:79-151):
- `role: str` - Message role (user/assistant/system/tool)
- `alias: Optional[str]` - Display name override
- `text: Optional[Union[str, Dict]]` - Main content (can be dict for structured)
- `attachments: List[Attachment]` - Images, files, etc.
- `tool_calls: List[ToolUse]` - **COMPLETED** tool executions with results
- `timestamp, tags, meta` - Metadata fields

**Prompt(Msg)** (dachi/core/_msg.py:153-207):
- Inherits all Msg fields
- Sampling parameters: `model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, seed`
- `system_message: Optional[str]` - System message override
- `reasoning_summary_request: Optional[bool]` - For reasoning models
- `tool_override: bool` - Whether to replace or extend tools
- `tools: Optional[List[BaseTool]]` - Available tools for this prompt
- `format_override: Optional[...]` - Output format control

**Resp(Msg)** (dachi/core/_msg.py:209-321):
- Inherits all Msg fields
- `model, finish_reason, usage, logprobs` - Generation metadata
- `thinking: Optional[Union[str, Dict]]` - Reasoning content (o1, etc.)
- `citations: Optional[List[Dict]]` - Source citations
- `tool_use: List[ToolUse]` - Tools LLM **WANTS TO EXECUTE** (not yet run)
- `id, out, choices` - Additional metadata
- `_raw: Dict` (private) - Raw API response for debugging

**DeltaResp** (dachi/core/_msg.py:323-366):
- **CRITICAL**: Contains ONLY incremental deltas for the current chunk
- `text: Optional[str]` - Text delta (NOT accumulated)
- `thinking: Optional[Union[str, Dict]]` - Reasoning delta (NOT accumulated)
- `citations: Optional[List[Dict]]` - Citation delta (NOT accumulated)
- `tool: Optional[str]` - Partial tool call JSON fragment
- `finish_reason, usage` - Set when streaming completes

**ToolUse** (dachi/core/_tool.py:252-297):
- `tool_id: str` - Unique ID for this tool call (maps to OpenAI's call_id)
- `id: Optional[str]` - Additional ID field
- `option: BaseTool` - The tool definition
- `inputs: BaseModel` - Validated input parameters (parsed from JSON arguments)
- `result: Any` - Execution result (if executed)
- `executed: bool` - Whether tool has been run

##### Incomplete/Broken Code

**CRITICAL MISSING FUNCTIONALITY**:

1. **dachi/proc/openai.py:164-168** - Tool extraction stub:
   ```python
   def extract_openai_tool_calls(message: dict, tools) -> list:
       """Extract tool calls from OpenAI response message"""
       pass  # ← COMPLETELY UNIMPLEMENTED!
   ```

2. **dachi/proc/openai.py:310, 517** - Both adapters call the broken stub above

3. **dachi/proc/openai.py:533-619** - OpenAIResp.from_streamed_result() is incomplete:
   ```python
   def from_streamed_result(self, result: t.Dict, ...):
       text =  # ← Incomplete assignment!
       citations = self.extract_citations(...)  # ← Method doesn't exist!
       thinking = extract_delta_thinking(...)   # ← Function doesn't exist!
       tool_use = extract_delta_tool_use(...)   # ← Function doesn't exist!
       # 551-618: All commented-out dead code
   ```

4. **tests/core/test_msg.py:114+** - Tests use old constructor:
   ```python
   return Resp(msg=_sample_msg())  # ← Resp doesn't take 'msg' param anymore!
   ```

##### What Works ✅

- Message base classes (Msg, Prompt, Resp, DeltaResp) - structure is sound
- Tool definition system (BaseTool, Tool, AsyncTool, ToolUse)
- OpenAIChat.from_result() - extracts text, model, usage, finish_reason correctly
- OpenAIChat.from_streamed_result() - accumulates text deltas correctly
- Message conversion (convert_messages) - handles attachments properly
- Format conversion helpers (build_openai_response_format, etc.)

##### What's Broken ❌

- Tool call extraction from non-streaming responses (stub function)
- Tool call extraction from streaming responses (doesn't exist)
- OpenAIResp.from_streamed_result() - barely started
- Tests in test_msg.py - use old Resp(msg=...) constructor

##### Critical Insights for Streaming Design

**IMPORTANT ORDERING** (from user feedback):
1. **DeltaResp must be created FIRST** from the current chunk/event
2. **Then create Resp** by accumulating DeltaResp with prev_resp
3. **Tool call completion detection** requires comparing current delta with previous accumulated state
4. **DeltaResp contains ONLY the delta** - not accumulated values
5. **Resp contains the ENTIRE accumulated response** - full text, full tool calls, etc.

This ordering is critical because:
- We need the delta to determine what changed
- We need both delta and previous state to detect tool call completion
- Tool calls are complete when we have valid JSON + name (requires checking accumulated arguments)

##### Summary

The framework structure is well-designed. The core issue is missing implementation:
1. No tool call extraction logic exists (stub only)
2. Streaming tool call reconstruction is completely missing
3. Responses API streaming handler is incomplete
4. Tests need updating for new Resp inheritance

The good news: Data structures are solid. We just need to implement extraction logic with proper delta → accumulated flow.

---

#### Subplan 1.2: API Contract Analysis
**Status**: ✅ COMPLETED
**Purpose**: Precisely understand what OpenAI APIs provide

**Activities**:
- Document complete Chat Completions response structure
- Document complete Responses API response structure
- Document streaming chunk formats for both APIs
- Map exact locations of: text, thinking, tool_calls, usage, etc.
- Identify commonalities and differences

**Acceptance Criteria**:
- [x] Exhaustive field mapping for both APIs (non-streaming)
- [x] Exhaustive event/delta mapping for both APIs (streaming)
- [x] Clear table showing differences between the two APIs
- [x] Reference to authoritative OpenAI documentation
- [x] No guesswork - everything must be verified

**Findings**:

##### Chat Completions API (Non-Streaming)

**Reference**: OpenAI Chat Completions API documentation

**Response Structure**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4",
  "system_fingerprint": "fp_xxx",
  "service_tier": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The answer is...",
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"NYC\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**Field Extraction Map**:
- `choices[0].message.role` → `Resp.role`
- `choices[0].message.content` → `Resp.text`
- `choices[0].message.tool_calls[]` → `Resp.tool_use[]` (convert to ToolUse)
- `choices[0].finish_reason` → `Resp.finish_reason`
- `choices[0].logprobs` → `Resp.logprobs`
- `id` → `Resp.id`
- `model` → `Resp.model`
- `usage` → `Resp.usage`
- `object, created, system_fingerprint, service_tier` → `Resp.meta`
- `choices[]` (all) → `Resp.choices` (for multi-choice support)
- Full response → `Resp._raw`

**Tool Call Structure**:
```json
{
  "id": "call_abc123",              // → ToolUse.tool_id
  "type": "function",               // (always "function")
  "function": {
    "name": "get_weather",          // → look up BaseTool, create ToolUse.option
    "arguments": "{\"location\":\"NYC\"}"  // → parse JSON → ToolUse.inputs
  }
}
```

##### Chat Completions API (Streaming)

**Reference**: OpenAI Chat Completions Streaming documentation

**Chunk Structure**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "gpt-4",
  "system_fingerprint": "fp_xxx",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",         // Only in first chunk
        "content": "The ",            // Incremental text fragment
        "tool_calls": [               // Incremental tool call fragments
          {
            "index": 0,               // Position in tool_calls array
            "id": "call_abc123",      // Only in first chunk for this tool
            "type": "function",       // Only in first chunk
            "function": {
              "name": "get_weather", // Only in first chunk
              "arguments": "{\"loc"  // Incremental JSON fragment
            }
          }
        ]
      },
      "finish_reason": null,         // null until final chunk
      "logprobs": null
    }
  ],
  "usage": null                       // null until final chunk
}
```

**Delta Extraction Map**:
- `choices[0].delta.content` → `DeltaResp.text` (just the fragment)
- `choices[0].delta.tool_calls[]` → accumulate into staging area
  - `tool_calls[i].index` → which tool call this belongs to
  - `tool_calls[i].id` → set on first chunk
  - `tool_calls[i].function.name` → set on first chunk
  - `tool_calls[i].function.arguments` → append fragments
- `choices[0].finish_reason` → `DeltaResp.finish_reason` (when present)
- `usage` → `DeltaResp.usage` (when present, final chunk)

**Streaming Reconstruction Requirements**:
1. Track multiple tool calls simultaneously by `index`
2. Accumulate `arguments` string fragments
3. Detect completion when arguments form valid JSON
4. Create ToolUse only when: `name` is set AND `arguments` is valid JSON

##### Responses API (Non-Streaming)

**Reference**: OpenAI Responses API documentation (openapi.yaml)

**Response Structure**:
```json
{
  "id": "resp_123",
  "object": "response",
  "created": 1677652288,
  "model": "o1-preview",
  "system_fingerprint": "fp_xxx",
  "service_tier": "default",
  "output": [
    {
      "type": "message",
      "id": "msg_123",
      "role": "assistant",
      "content": "The answer is..."
    },
    {
      "type": "function_call",
      "id": "fc_123",
      "call_id": "call_abc123",
      "name": "get_weather",
      "arguments": "{\"location\":\"NYC\"}",
      "status": "completed"
    }
  ],
  "reasoning": "Let me think about this...",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30,
    "reasoning_tokens": 5
  }
}
```

**Field Extraction Map**:
- Extract message from `output[]` where `type == "message"`:
  - `output[i].role` → `Resp.role`
  - `output[i].content` → `Resp.text`
- Extract tool calls from `output[]` where `type == "function_call"`:
  - `output[i].call_id` → `ToolUse.tool_id`
  - `output[i].name` → look up BaseTool → `ToolUse.option`
  - `output[i].arguments` → parse JSON → `ToolUse.inputs`
- `reasoning` → `Resp.thinking`
- `choices[0].finish_reason` → `Resp.finish_reason`
- `choices[0].logprobs` → `Resp.logprobs`
- `id` → `Resp.id`
- `model` → `Resp.model`
- `usage` → `Resp.usage`
- `object, created, system_fingerprint, service_tier` → `Resp.meta`
- `choices[]` (all) → `Resp.choices`
- Full response → `Resp._raw`

##### Responses API (Streaming)

**Reference**: OpenAI Responses API streaming events

**Event Types**:
```json
// Event 1: Output item appears
{
  "type": "response.output_item.added",
  "item": {
    "type": "function_call",
    "id": "fc_123",
    "call_id": "call_abc123",
    "name": "get_weather",
    "arguments": ""  // May be empty initially
  }
}

// Event 2: Arguments stream in
{
  "type": "response.function_call_arguments.delta",
  "item_id": "fc_123",
  "delta": "{\"loc"  // Fragment to append
}

// Event 3: Arguments finalized
{
  "type": "response.function_call_arguments.done",
  "item_id": "fc_123",
  "arguments": "{\"location\":\"NYC\"}"  // Complete JSON
}

// Event 4: Output item done
{
  "type": "response.output_item.done",
  "item": {
    "type": "function_call",
    "id": "fc_123",
    "call_id": "call_abc123",
    "name": "get_weather",
    "arguments": "{\"location\":\"NYC\"}",
    "status": "completed"
  }
}

// Text content delta (for message output items)
{
  "type": "response.text.delta",
  "delta": "The "
}

// Reasoning delta (o1 models)
{
  "type": "response.reasoning.delta",
  "delta": "Let me "
}
```

**Event Processing Map**:
- `response.output_item.added` (type=function_call):
  - Extract `item.call_id` → stage with this ID
  - Extract `item.name` → store in staging
  - Extract `item.arguments` → initialize accumulator
- `response.function_call_arguments.delta`:
  - Extract `item_id` → find staging entry
  - Extract `delta` → append to arguments accumulator
- `response.function_call_arguments.done`:
  - Extract `item_id` → find staging entry
  - Extract `arguments` → replace accumulator (final value)
  - Validate JSON → if valid, create ToolUse
- `response.text.delta`:
  - Extract `delta` → `DeltaResp.text`
- `response.reasoning.delta`:
  - Extract `delta` → `DeltaResp.thinking`

**Streaming Reconstruction Requirements**:
1. Maintain staging dict keyed by `item_id`
2. Track: `call_id`, `name`, accumulated `arguments`
3. Detect completion on `.done` event OR when arguments validate as JSON
4. Create ToolUse when: `name` exists AND `arguments` is valid JSON

##### API Comparison Table

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| **Non-streaming response object** | `chat.completion` | `response` |
| **Streaming chunk object** | `chat.completion.chunk` | Event stream |
| **Text location (non-stream)** | `choices[0].message.content` | `output[].content` (where type=message) |
| **Text delta (stream)** | `choices[0].delta.content` | `response.text.delta` event |
| **Thinking/reasoning** | Not available | `reasoning` field / `response.reasoning.delta` event |
| **Tool calls (non-stream)** | `choices[0].message.tool_calls[]` | `output[]` where type=function_call |
| **Tool call ID field** | `tool_calls[].id` | `function_call.call_id` |
| **Tool call streaming** | `delta.tool_calls[i]` with index | Separate events by item_id |
| **Tool args accumulation** | By index position | By item_id |
| **Completion detection** | Valid JSON check | `.done` event + valid JSON |
| **Usage tokens** | `usage{}` | `usage{}` (includes reasoning_tokens) |
| **Role location** | `message.role` | `output[].role` |
| **Finish reason** | `choices[0].finish_reason` | `choices[0].finish_reason` |

##### Key Differences for Implementation

1. **Structure**: Chat uses `message.tool_calls[]`, Responses uses `output[]` items
2. **Tool ID**: Chat uses `id`, Responses uses `call_id`
3. **Streaming**: Chat uses indexed deltas, Responses uses event types + item_id
4. **Thinking**: Only in Responses API
5. **Both require**: JSON validation for tool argument completion

##### Authoritative References

- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat
- OpenAI Responses API: https://github.com/openai/openai-openapi (openapi.yaml)
- Streaming tool calls reconstruction: Provided in user's API overview document

---

#### Subplan 1.3: Requirements Definition
**Status**: ✅ COMPLETED
**Purpose**: Define exactly what we must achieve

**Activities**:
- Define what data MUST be extracted from each API
- Define what data MUST be stored in Dachi messages
- Define invariants (e.g., tool_use = pending, tool_calls = completed)
- Define streaming accumulation requirements
- Define error handling requirements

**Acceptance Criteria**:
- [x] Clear, testable requirements for data extraction
- [x] Clear invariants that code must maintain
- [x] Requirements are complete (cover all APIs, all modes)
- [x] Requirements are minimal (no gold-plating)
- [x] Requirements are verifiable (we can test them)

**Findings**:

##### Functional Requirements

**FR1: Non-Streaming Extraction (Chat Completions)**
- MUST extract: role, text (content), tool_calls, finish_reason, id, model, usage, logprobs
- MUST create ToolUse objects from tool_calls array
- MUST validate tool call arguments are valid JSON before creating ToolUse
- MUST look up BaseTool by name from available tools
- MUST handle missing/null content gracefully
- MUST store metadata (object, created, system_fingerprint, service_tier) in Resp.meta
- MUST store full response in Resp._raw

**FR2: Non-Streaming Extraction (Responses API)**
- MUST extract from output[] array:
  - Message items (type="message"): role, content
  - Function call items (type="function_call"): call_id, name, arguments
- MUST extract reasoning field → Resp.thinking
- MUST extract choices, usage, id, model, finish_reason, logprobs
- MUST create ToolUse objects from function_call items
- MUST validate arguments are valid JSON
- MUST look up BaseTool by name
- MUST store metadata in Resp.meta
- MUST store full response in Resp._raw

**FR3: Streaming Delta Extraction (Chat Completions)**
- MUST create DeltaResp FIRST from current chunk
- MUST extract delta.content → DeltaResp.text (fragment only)
- MUST extract delta.tool_calls[] for staging
- MUST track tool calls by index position
- MUST accumulate arguments fragments
- MUST detect tool call completion (valid JSON check)
- MUST create Resp by accumulating DeltaResp + prev_resp
- MUST preserve prev_resp.role if delta.role is None
- MUST handle finish_reason and usage in final chunk

**FR4: Streaming Delta Extraction (Responses API)**
- MUST create DeltaResp FIRST from current event
- MUST handle event types:
  - `response.text.delta` → DeltaResp.text
  - `response.reasoning.delta` → DeltaResp.thinking
  - `response.output_item.added` (function_call) → stage tool call
  - `response.function_call_arguments.delta` → accumulate arguments
  - `response.function_call_arguments.done` → finalize tool call
- MUST track tool calls by item_id
- MUST accumulate arguments per item_id
- MUST detect completion on .done event
- MUST create Resp by accumulating DeltaResp + prev_resp
- MUST validate JSON before creating ToolUse

**FR5: Tool Call to ToolUse Conversion**
- MUST extract tool_id from:
  - Chat: `tool_calls[].id`
  - Responses: `function_call.call_id`
- MUST look up BaseTool from available tools by name
- MUST parse arguments JSON string into dict
- MUST validate with tool's input_model to create ToolUse.inputs
- MUST handle lookup failures gracefully (tool not found)
- MUST handle JSON parse failures gracefully
- MUST handle validation failures gracefully

##### Data Storage Requirements

**DSR1: Resp Object (Non-Streaming)**
- MUST contain complete, final values (not deltas)
- MUST populate: role, text, tool_use, finish_reason, id, model, usage
- MUST populate thinking (Responses API only)
- MUST store raw response in _raw
- MUST store metadata in meta
- tool_use contains ToolUse objects ready to execute

**DSR2: Resp Object (Streaming - Accumulated)**
- MUST contain complete accumulated text (not just latest delta)
- MUST contain complete accumulated thinking (not just latest delta)
- MUST contain all completed ToolUse objects (accumulated from all chunks)
- MUST preserve all fields from DSR1
- MUST maintain accumulation across all chunks

**DSR3: DeltaResp Object (Streaming - Delta Only)**
- MUST contain ONLY incremental values for current chunk
- text: just the fragment added in this chunk
- thinking: just the reasoning fragment added in this chunk
- tool: partial JSON fragment for tool arguments (if any)
- MUST NOT contain accumulated values
- MUST be created BEFORE Resp in streaming flow

**DSR4: ToolUse Object**
- tool_id: MUST be unique identifier from API
- option: MUST be valid BaseTool looked up by name
- inputs: MUST be validated BaseModel instance
- result: initially None (set when executed)
- executed: initially False

##### Invariants (MUST ALWAYS HOLD)

**INV1: Tool Use vs Tool Calls**
- `Resp.tool_use`: Contains tools LLM wants to execute (not yet run, result=None, executed=False)
- `Resp.tool_calls`: Contains tools that have been executed (result set, executed=True)
- These MUST be mutually exclusive (a ToolUse moves from tool_use to tool_calls on execution)

**INV2: Streaming Delta vs Accumulated**
- DeltaResp contains ONLY deltas (fragments)
- Resp contains ONLY accumulated complete values
- Never mix: DeltaResp.text is NOT cumulative

**INV3: Tool Call Completion**
- ToolUse MUST ONLY be created when:
  - Name exists AND
  - Arguments are valid JSON AND
  - JSON can be parsed into dict AND
  - BaseTool with that name exists
- Incomplete tool calls MUST remain in staging (not in Resp.tool_use)

**INV4: Streaming Processing Order**
- MUST create DeltaResp first
- MUST create Resp second (using DeltaResp + prev_resp)
- MUST detect tool completion using both delta and accumulated state

**INV5: Text Accumulation**
- Streaming: `Resp.text = (prev_resp.text or "") + DeltaResp.text`
- Same for thinking field
- MUST handle None values gracefully (treat as empty string)

##### Error Handling Requirements

**EH1: Missing Tools**
- If BaseTool not found by name: Log warning, skip creating ToolUse, continue processing
- MUST NOT crash or halt extraction
- MUST NOT corrupt other tool calls

**EH2: Invalid JSON**
- If arguments fail JSON parse: Skip creating ToolUse, continue processing
- In streaming: Keep accumulating until valid OR stream ends
- MUST NOT crash

**EH3: Validation Failures**
- If input_model validation fails: Log error, skip creating ToolUse, continue
- MUST provide useful error message for debugging
- MUST NOT crash

**EH4: Missing/Null Fields**
- If content/text is null: Use empty string
- If role is missing in delta: Use prev_resp.role or default "assistant"
- If usage is null: Use empty dict
- MUST handle all null/None cases gracefully

**EH5: Malformed Responses**
- If response structure doesn't match expected: Extract what's possible, log warning
- MUST NOT crash on unexpected structure
- MUST return partial Resp with what was extractable

##### Streaming-Specific Requirements

**SR1: State Management**
- Staging area for incomplete tool calls MUST be stored in prev_resp._raw
- Use keys like: `_tool_calls_staging_chat` (by index) or `_tool_calls_staging_resp` (by item_id)
- MUST persist staging across chunks
- MUST clean up staging when tool calls complete

**SR2: Tool Call Staging Structure (Chat Completions)**
```python
_tool_calls_staging_chat = {
    0: {  # index position
        "id": "call_abc123",
        "name": "get_weather",
        "arguments": "{\"location\":\"NYC\"}"  # accumulated so far
    },
    1: { ... }
}
```

**SR3: Tool Call Staging Structure (Responses API)**
```python
_tool_calls_staging_resp = {
    "fc_123": {  # item_id
        "call_id": "call_abc123",
        "name": "get_weather",
        "arguments": "{\"location\":\"NYC\"}"  # accumulated so far
    }
}
```

**SR4: Completion Detection**
- Chat: After each chunk, check each staged tool if arguments form valid JSON
- Responses: Check on `.done` event AND validate JSON
- When complete: Create ToolUse, add to Resp.tool_use, remove from staging

**SR5: Final Chunk Handling**
- MUST process all remaining staged tool calls
- MUST attempt to create ToolUse for any with valid JSON
- MUST handle finish_reason and usage from final chunk

##### Performance Requirements

**PR1: No Unnecessary Parsing**
- Parse JSON only when validation needed (not every chunk)
- Use simple string checks first (starts with "{", etc.)

**PR2: Memory Management**
- Clean up staging when tool calls complete
- Don't accumulate unbounded state
- Limit staging dict size (reasonable limit: 100 concurrent tool calls)

##### Testing Requirements

**TR1: All APIs and Modes**
- Test Chat Completions non-streaming
- Test Chat Completions streaming
- Test Responses API non-streaming
- Test Responses API streaming
- Test both with and without tool calls

**TR2: Edge Cases**
- Empty content/text
- Null fields
- Invalid JSON
- Missing tools
- Multiple concurrent tool calls (streaming)
- Tool call spans many chunks
- Malformed responses

**TR3: Validation**
- Verify invariants hold
- Verify DeltaResp contains only deltas
- Verify Resp contains accumulated values
- Verify staging cleanup
- Verify error handling doesn't crash

##### Summary

These requirements ensure:
1. **Correctness**: Proper extraction from both APIs
2. **Robustness**: Graceful error handling
3. **Simplicity**: Clear separation of concerns
4. **Testability**: All requirements are verifiable
5. **Maintainability**: Clear invariants and contracts

---

### Phase 2: Architecture Design

#### Subplan 2.1: Data Structure Design
**Status**: ✅ COMPLETED
**Purpose**: Decide what goes where in our message objects

**Activities**:
- Finalize Resp vs DeltaResp field usage
- Design streaming accumulation strategy
- Define when/how ToolUse objects are created
- Design error handling for malformed data

**Acceptance Criteria**:
- [x] Every piece of data has a clear home
- [x] No redundancy (don't store same thing twice)
- [x] Clear separation: deltas vs accumulated values
- [x] Streaming state management is simple
- [x] Design is extensible for future APIs

**Findings**:

##### High-Level Processing Steps

**Non-Streaming Processing (Both APIs)**:
```
Step 1: Receive complete API response (dict)
Step 2: Extract simple scalar fields (role, id, model, finish_reason, usage, etc.)
Step 3: Extract text content (different location per API)
Step 4: Extract thinking/reasoning (Responses API only)
Step 5: Extract tool calls (different structure per API)
Step 6: For each tool call:
        - Validate arguments are valid JSON
        - Look up BaseTool by name
        - Create ToolUse object (if successful)
Step 7: Store metadata and raw response
Step 8: Return Resp object
```

**Streaming Processing (Both APIs)**:
```
Step 1: Receive chunk/event (dict)
Step 2: Get staging area from prev_resp (if exists)

Step 3: CREATE DELTARESP FIRST
        3a: Extract text delta
        3b: Extract thinking delta (Responses only)
        3c: Extract tool call deltas → update staging
        3d: Extract finish_reason, usage (if present)
        3e: Create DeltaResp object with ONLY deltas

Step 4: CREATE RESP BY ACCUMULATION
        4a: Accumulate text (prev + delta)
        4b: Accumulate thinking (prev + delta)
        4c: Copy completed tool calls from prev
        4d: Copy other fields from prev or use delta values
        4e: Detect tool call completions in staging
        4f: Create ToolUse for completed tools, add to Resp
        4g: Store updated staging in new Resp

Step 5: Return (Resp, DeltaResp) tuple
```

##### Natural Processing Boundaries

Based on these steps, we can identify clear boundaries:

**Boundary 1: Delta Extraction** (Step 3)
- Input: Raw chunk/event
- Output: DeltaResp object
- Responsibility: Extract ONLY the new fragments from this chunk
- Simple: No accumulation logic, no tool completion detection

**Boundary 2: Staging Update** (Step 3c)
- Input: Tool call deltas from chunk + existing staging
- Output: Updated staging dict
- Responsibility: Append fragments, track tool call state
- Simple: Just string concatenation and dict updates

**Boundary 3: Accumulation** (Step 4a-4d)
- Input: prev_resp + DeltaResp
- Output: New accumulated values
- Responsibility: Combine previous state with deltas
- Simple: String concatenation, preserve existing values

**Boundary 4: Tool Completion Detection** (Step 4e)
- Input: Staging dict
- Output: List of completed tool call dicts
- Responsibility: Check which staged tools have valid JSON
- Simple: Iterate staging, try JSON parse, return complete ones

**Boundary 5: ToolUse Creation** (Step 4f)
- Input: Completed tool call dict + available tools
- Output: ToolUse object (or None if failed)
- Responsibility: Look up tool, parse JSON, validate, create ToolUse
- Simple: Single tool call → single ToolUse, no iteration

##### Data Flow Diagram

```
NON-STREAMING:
API Response → Extract Fields → Extract Tool Calls → Create ToolUse → Resp

STREAMING:
                    ┌─────────────────┐
API Chunk/Event ───→│ Extract Deltas  │──→ DeltaResp
                    └─────────────────┘
                            │
                            ↓
                    ┌─────────────────┐
prev_resp ─────────→│ Update Staging  │──→ Updated Staging
                    └─────────────────┘
                            │
                            ↓
                    ┌─────────────────┐
prev_resp + Delta ─→│ Accumulate Text │──→ Accumulated Text
                    └─────────────────┘
                            │
                            ↓
                    ┌─────────────────┐
Updated Staging ───→│ Detect Complete │──→ Completed Tool Dicts
                    └─────────────────┘
                            │
                            ↓
                    ┌─────────────────┐
Complete + Tools ──→│ Create ToolUse  │──→ ToolUse Objects
                    └─────────────────┘
                            │
                            ↓
                        New Resp
```

##### Streaming State Management Design

**What State Lives Where**:

1. **DeltaResp** (ephemeral, not persisted):
   - `text`: Text fragment from this chunk
   - `thinking`: Thinking fragment from this chunk
   - `tool`: Arguments fragment (if any) - just for user visibility
   - `finish_reason`, `usage`: If present in this chunk

2. **Resp** (accumulated, persisted across chunks):
   - `text`: FULL accumulated text
   - `thinking`: FULL accumulated thinking
   - `tool_use`: List of COMPLETED ToolUse objects
   - `_raw`: Dict containing:
     - Normal raw response fields
     - `_staging_chat` or `_staging_resp`: Incomplete tool calls

3. **Staging Dict Structure** (CANONICAL FORMAT):

   Both APIs use the same canonical structure, just keyed differently:

   **Chat Completions**:
   ```python
   _staging_chat = {
       0: {  # Key: index (int)
           "id": "call_abc123",
           "type": "function",
           "function": {
               "name": "get_weather",
               "arguments": "{\"location\":"  # Accumulated so far
           },
           "_complete": False  # Internal flag for completion tracking
       },
       1: { ... }
   }
   ```

   **Responses API**:
   ```python
   _staging_resp = {
       "fc_123": {  # Key: item_id (str)
           "id": "call_abc123",  # from call_id field
           "type": "function",
           "function": {
               "name": "get_weather",
               "arguments": "{\"location\":"  # Accumulated so far
           },
           "_complete": False  # Internal flag for completion tracking
       },
       "fc_456": { ... }
   }
   ```

   **Benefits of Canonical Format**:
   - Same completion detection logic works for both APIs
   - Easy to convert to ToolUse (just extract from canonical format)
   - `_complete` flag prevents re-processing completed tools
   - Matches OpenAI's internal structure

**Staging Lifecycle**:
1. Initialize: Empty dict on first chunk
2. Add entry: When tool call first appears (has id/call_id and name)
3. Update entry: Append to arguments on each delta
4. Detect complete: Try JSON parse, check if valid
5. Remove entry: When moved to Resp.tool_use as ToolUse
6. Persist: Store in Resp._raw between chunks

##### ToolUse Creation Decision Points

**When to Create ToolUse** (ALL must be true):
1. Name exists (not None, not empty)
2. Arguments string is valid JSON (parse succeeds)
3. Parsed JSON is a dict
4. BaseTool with that name is available
5. Input validation succeeds (tool.input_model(**args_dict))

**When NOT to Create ToolUse**:
- Name missing → keep in staging
- Arguments not valid JSON yet → keep in staging
- Tool not found → log warning, remove from staging
- Validation fails → log error, remove from staging

**Error Handling Strategy**:
- Invalid JSON: Stay in staging (streaming) or skip (non-streaming)
- Missing tool: Log warning, skip, continue
- Validation error: Log error with details, skip, continue
- Never crash, always return partial Resp

##### Field-by-Field Storage Design

**Resp Object Fields**:
```python
# Core message fields (from Msg)
role: str                                    # From API or prev_resp
text: str | Dict | None                      # Accumulated in streaming
tool_calls: List[ToolUse]                    # Executed tools (not populated by adapters)

# Response-specific fields
model: str | None                            # From API
finish_reason: str | None                    # From API
usage: Dict[str, Any]                        # From API
logprobs: Dict | None                        # From API
thinking: str | Dict | None                  # Accumulated in streaming (Responses only)
citations: List[Dict] | None                 # From API (future)
tool_use: List[ToolUse]                      # Created from completed tools
id: str | None                               # From API
out: Any                                     # Processed by ToOut (not adapter concern)
choices: List[Dict] | None                   # From API (multi-choice)
meta: Dict[str, Any]                         # Metadata (object, created, etc.)
_raw: Dict                                   # Full response + staging
```

**DeltaResp Object Fields**:
```python
text: str | None                             # Fragment only
thinking: str | Dict | None                  # Fragment only
citations: List[Dict] | None                 # Fragment only (future)
tool: str | None                             # Arguments fragment (user visibility)
finish_reason: str | None                    # Only in final chunk
usage: Dict | None                           # Only in final chunk
```

##### Simplification Principles

1. **One Responsibility Per Boundary**:
   - Delta extraction: Just pull out fragments
   - Staging update: Just append strings
   - Accumulation: Just concatenate
   - Completion detection: Just validate JSON
   - ToolUse creation: Just create object

2. **No Hidden State**:
   - Everything flows through prev_resp
   - No module-level variables
   - No side effects in helpers

3. **Fail Independently**:
   - One tool call failure doesn't affect others
   - JSON parse failure doesn't crash extraction
   - Missing tool doesn't stop processing

4. **Easy to Test**:
   - Each boundary has clear inputs/outputs
   - Can test with simple dicts/strings
   - No mocking needed for most tests

5. **Minimal Changes to Existing Structure**:
   - Resp and DeltaResp already exist and are correct
   - ToolUse already exists and is correct
   - Just need to populate them properly

##### Summary

The design is simple because:
- **5 clear boundaries**: Delta extraction, staging update, accumulation, completion detection, ToolUse creation
- **Clear data homes**: DeltaResp for fragments, Resp for accumulated, staging in _raw
- **Simple lifecycle**: Add to staging → accumulate → detect complete → create ToolUse → remove from staging
- **Independent failures**: Each tool call, each field, each operation can fail without affecting others
- **No architectural changes**: Use existing data structures correctly

---

#### Subplan 2.2: Module Boundary Design
**Status**: ✅ COMPLETED
**Purpose**: Define clear responsibilities and interfaces

**Activities**:
- Define adapter responsibilities
- Define helper function responsibilities
- Design state flow (how prev_resp is used)
- Define dependency relationships

**Acceptance Criteria**:
- [x] Single Responsibility Principle maintained
- [x] Loose coupling between components
- [x] High cohesion within components
- [x] No circular dependencies
- [x] Clear interfaces with well-defined inputs/outputs
- [x] Stateless where possible

**Findings**:

##### Module Organization

We'll keep the existing module structure with minimal changes:

**Existing Modules** (no changes):
- `dachi/core/_msg.py` - Msg, Resp, DeltaResp, ToolUse (already correct)
- `dachi/core/_tool.py` - BaseTool, Tool, AsyncTool, ToolUse (already correct)
- `dachi/proc/_ai.py` - LangEngine base class, helper functions (already correct)

**Module to Update**:
- `dachi/proc/openai.py` - OpenAIChat, OpenAIResp adapters (add complete implementations)

**No New Modules Needed**: All helper functions will live in `openai.py` as private functions

##### Responsibility Assignment

**Adapters (OpenAIChat, OpenAIResp) - Orchestration**:
- Responsibility: Coordinate the extraction process
- What they do:
  - Call helper functions in correct sequence
  - Handle API-specific differences
  - Return properly constructed Resp/DeltaResp objects
- What they DON'T do:
  - Direct dict traversal
  - String manipulation
  - JSON parsing
  - Tool lookup

**Helper Functions - Single Operations**:
- Responsibility: One specific data transformation
- What they do:
  - Take simple inputs (dicts, strings)
  - Perform one operation
  - Return simple outputs (strings, dicts, objects)
- What they DON'T do:
  - Call other helpers
  - Maintain state
  - Make decisions about control flow

##### Function Mapping to Boundaries

**Boundary 1: Delta Extraction**
- Functions needed:
  - `_extract_text_delta_chat(chunk: dict) -> str | None`
  - `_extract_text_delta_resp(event: dict) -> str | None`
  - `_extract_thinking_delta_resp(event: dict) -> str | None`
  - `_extract_tool_deltas_chat(chunk: dict) -> list[dict]`
  - `_extract_tool_deltas_resp(event: dict) -> dict | None`

**Boundary 2: Staging Update**
- Functions needed:
  - `_update_staging_chat(staging: dict, tool_deltas: list[dict]) -> dict`
  - `_update_staging_resp(staging: dict, event_data: dict) -> dict`

**Boundary 3: Accumulation**
- Functions needed:
  - `_accumulate_text(prev: str | None, delta: str | None) -> str`
  - (Same function works for thinking)

**Boundary 4: Tool Completion Detection**
- Functions needed:
  - `_detect_complete_tools(staging: dict) -> tuple[list[dict], dict]`
    - Returns: (completed_tools, remaining_staging)

**Boundary 5: ToolUse Creation**
- Functions needed:
  - `_create_tool_use(tool_dict: dict, available_tools: list[BaseTool]) -> ToolUse | None`

**Non-Streaming Extraction**:
- Functions needed:
  - `_extract_tool_calls_chat(message: dict) -> list[dict]`
  - `_extract_tool_calls_resp(output: list[dict]) -> list[dict]`

**Utility Functions**:
- `_is_valid_json(s: str) -> bool`
- `_get_staging(resp: Resp | None, key: str) -> dict`
- `_store_staging(resp: Resp, key: str, staging: dict) -> None`

##### Interface Specifications

**Delta Extraction Functions**:
```python
def _extract_text_delta_chat(chunk: dict) -> str | None:
    """Extract text delta from Chat Completions chunk.

    Args:
        chunk: Raw API chunk (choices[0].delta...)

    Returns:
        Text fragment or None if not present

    Example:
        >>> chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        >>> _extract_text_delta_chat(chunk)
        "Hello"
    """
```

**Staging Update Functions**:
```python
def _update_staging_chat(staging: dict, tool_deltas: list[dict]) -> dict:
    """Update staging with tool call deltas from Chat Completions.

    Args:
        staging: Current staging dict {index: {id, name, arguments}}
        tool_deltas: Tool call deltas from chunk

    Returns:
        Updated staging dict (new copy, not mutated)

    Example:
        >>> staging = {0: {"id": "call_1", "name": "foo", "arguments": "{"}}
        >>> deltas = [{"index": 0, "function": {"arguments": '"x"'}}]
        >>> _update_staging_chat(staging, deltas)
        {0: {"id": "call_1", "name": "foo", "arguments": '{"x"'}}
    """
```

**Accumulation Functions**:
```python
def _accumulate_text(prev: str | None, delta: str | None) -> str:
    """Accumulate text by concatenating previous and delta.

    Args:
        prev: Previous accumulated text (or None)
        delta: New text fragment (or None)

    Returns:
        Concatenated text (empty string if both None)

    Example:
        >>> _accumulate_text("Hello", " world")
        "Hello world"
        >>> _accumulate_text(None, "Hello")
        "Hello"
    """
```

**Completion Detection Functions**:
```python
def _detect_complete_tools(staging: dict) -> tuple[list[dict], dict]:
    """Find tools with valid JSON arguments and remove from staging.

    Args:
        staging: Current staging dict

    Returns:
        Tuple of (completed_tools, remaining_staging)
        - completed_tools: List of tool dicts with valid JSON
        - remaining_staging: Updated staging without completed tools

    Example:
        >>> staging = {
        ...     0: {"id": "c1", "name": "foo", "arguments": '{"x": 1}'},
        ...     1: {"id": "c2", "name": "bar", "arguments": '{"y"}
        ... }
        >>> complete, remaining = _detect_complete_tools(staging)
        >>> len(complete)
        1
        >>> len(remaining)
        1
    """
```

**ToolUse Creation Functions**:
```python
def _create_tool_use(tool_dict: dict, available_tools: list[BaseTool]) -> ToolUse | None:
    """Create ToolUse object from tool call dict.

    Args:
        tool_dict: Dict with keys: id/call_id, name, arguments
        available_tools: List of available BaseTool objects

    Returns:
        ToolUse object or None if creation fails

    Side effects:
        - Logs warning if tool not found
        - Logs error if validation fails

    Example:
        >>> tool_dict = {"id": "c1", "name": "get_weather", "arguments": '{"location": "NYC"}'}
        >>> tools = [weather_tool]
        >>> tool_use = _create_tool_use(tool_dict, tools)
        >>> tool_use.tool_id
        "c1"
    """
```

##### Adapter Method Signatures

**OpenAIChat**:
```python
class OpenAIChat(LangEngine):
    def from_result(self, output: dict, messages: Msg | BaseDialog) -> Resp:
        """Convert Chat Completions response to Resp.

        Steps:
        1. Extract tool calls using _extract_tool_calls_chat()
        2. Create ToolUse objects using _create_tool_use()
        3. Extract other fields directly
        4. Construct and return Resp
        """

    def from_streamed_result(
        self,
        output: dict,
        messages: Msg | BaseDialog,
        prev_resp: Resp | None
    ) -> tuple[Resp, DeltaResp]:
        """Convert Chat Completions chunk to (Resp, DeltaResp).

        Steps:
        1. Extract deltas (text, tool_calls)
        2. Create DeltaResp
        3. Get staging from prev_resp
        4. Update staging with tool deltas
        5. Accumulate text
        6. Detect complete tools
        7. Create ToolUse for complete tools
        8. Construct Resp with accumulated values + staging
        9. Return (Resp, DeltaResp)
        """
```

**OpenAIResp**:
```python
class OpenAIResp(LangEngine):
    def from_result(self, output: dict, messages: Msg | BaseDialog) -> Resp:
        """Convert Responses API response to Resp.

        Steps:
        1. Extract message from output[] (type=message)
        2. Extract tool calls from output[] (type=function_call)
        3. Create ToolUse objects using _create_tool_use()
        4. Extract thinking, usage, etc.
        5. Construct and return Resp
        """

    def from_streamed_result(
        self,
        output: dict,
        messages: Msg | BaseDialog,
        prev_resp: Resp | None
    ) -> tuple[Resp, DeltaResp]:
        """Convert Responses API event to (Resp, DeltaResp).

        Steps:
        1. Extract deltas based on event type
        2. Create DeltaResp
        3. Get staging from prev_resp
        4. Update staging if tool event
        5. Accumulate text/thinking
        6. Detect complete tools
        7. Create ToolUse for complete tools
        8. Construct Resp with accumulated values + staging
        9. Return (Resp, DeltaResp)
        """
```

##### Data Flow Through Modules

```
INPUT: API Response/Chunk
    ↓
ADAPTER METHOD (from_result or from_streamed_result)
    ↓
HELPER FUNCTIONS (called in sequence):
    _extract_*() → raw values
    _update_staging() → staging dict
    _accumulate_*() → accumulated values
    _detect_complete_tools() → complete tool dicts
    _create_tool_use() → ToolUse objects
    ↓
ADAPTER METHOD (construct Resp/DeltaResp)
    ↓
OUTPUT: Resp (and DeltaResp for streaming)
```

##### State Flow Design

**Non-Streaming**:
```
API Response → Adapter → Helpers → Resp
(No state to manage)
```

**Streaming**:
```
Chunk 1: None → Adapter → Helpers → Resp₁, DeltaResp₁
                                      ↓ (staging in Resp₁._raw)
Chunk 2: Resp₁ → Adapter → Helpers → Resp₂, DeltaResp₂
                                      ↓ (staging in Resp₂._raw)
Chunk 3: Resp₂ → Adapter → Helpers → Resp₃, DeltaResp₃
```

**Staging Access Pattern**:
```python
# In adapter method:
staging = _get_staging(prev_resp, "_staging_chat")  # Returns {} if None
# ... update staging ...
_store_staging(new_resp, "_staging_chat", updated_staging)
```

##### Dependency Graph

```
Adapters
    ↓ (calls)
Helper Functions
    ↓ (uses)
Core Data Structures (Msg, Resp, DeltaResp, ToolUse)
    ↓ (uses)
Tool System (BaseTool, get_tool_function)
```

**No Circular Dependencies**:
- Helpers don't call adapters
- Helpers don't call other helpers
- Adapters don't depend on each other
- Core structures don't depend on anything (pure data)

##### Error Handling Distribution

**Helpers**:
- Return None on failure
- Log warnings/errors
- Never raise exceptions

**Adapters**:
- Handle None returns gracefully
- Always return valid Resp/DeltaResp
- Never crash

##### Testing Strategy by Module

**Helper Function Tests** (unit tests):
- Input: Simple dicts/strings
- Output: Verify correct extraction/transformation
- No mocking needed
- Fast, isolated

**Adapter Method Tests** (integration tests):
- Input: Real API response structures
- Output: Verify complete Resp/DeltaResp
- May mock tool lookup
- Test full flow

##### Summary

**Module Boundaries**:
- Adapters: Orchestration (call helpers, construct objects)
- Helpers: Single operations (extract, update, accumulate, detect, create)
- No new modules needed
- All code in `dachi/proc/openai.py`

**Key Principles**:
- ✅ Single Responsibility: Each function does one thing
- ✅ Loose Coupling: Helpers don't know about each other
- ✅ High Cohesion: Related functions grouped in same module
- ✅ No Circular Dependencies: Clear dependency hierarchy
- ✅ Clear Interfaces: Typed inputs/outputs
- ✅ Mostly Stateless: Only state is prev_resp (passed explicitly)

---

#### Subplan 2.3: Extraction Pipeline Design
**Status**: ✅ COMPLETED
**Purpose**: Define step-by-step process for data extraction

**Activities**:
- Design non-streaming extraction flow
- Design streaming accumulation flow
- Design tool call matching (name → BaseTool)
- Design JSON parsing & validation approach

**Acceptance Criteria**:
- [x] Step-by-step algorithm is clear and complete
- [x] Error paths are defined
- [x] Streaming and non-streaming paths are consistent
- [x] Algorithm is simple to implement
- [x] Algorithm handles edge cases

**Findings**:

##### Non-Streaming Extraction Pipeline

**OpenAIChat.from_result() Algorithm**:
```
1. Get available tools from messages
   tools = extract_tools_from_messages(messages)

2. Extract choice and message
   choice = output.get("choices", [{}])[0]
   message = choice.get("message", {})

3. Extract tool call dicts
   tool_dicts = _extract_tool_calls_chat(message)

4. Create ToolUse objects
   tool_use = []
   for tool_dict in tool_dicts:
       tu = _create_tool_use(tool_dict, tools)
       if tu is not None:
           tool_use.append(tu)

5. Extract simple fields
   role = message.get("role", "assistant")
   text = message.get("content") or ""
   finish_reason = choice.get("finish_reason")
   ... etc

6. Extract metadata
   meta = extract_commonly_useful_meta(output)

7. Construct Resp
   resp = Resp(
       role=role,
       text=text,
       tool_use=tool_use,
       finish_reason=finish_reason,
       ...
   )
   resp.raw = output
   resp.meta.update(meta)

8. Return resp
```

**OpenAIResp.from_result() Algorithm**:
```
1. Get available tools from messages
   tools = extract_tools_from_messages(messages)

2. Extract output array
   output_items = output.get("output", [])

3. Find message item
   message_item = first item where type == "message"
   role = message_item.get("role", "assistant")
   text = message_item.get("content") or ""

4. Extract tool calls
   tool_dicts = _extract_tool_calls_resp(output_items)

5. Create ToolUse objects
   tool_use = []
   for tool_dict in tool_dicts:
       tu = _create_tool_use(tool_dict, tools)
       if tu is not None:
           tool_use.append(tu)

6. Extract thinking and other fields
   thinking = output.get("reasoning")
   choice = output.get("choices", [{}])[0]
   finish_reason = choice.get("finish_reason")
   ... etc

7. Extract metadata
   meta = extract_commonly_useful_meta(output)

8. Construct Resp
   resp = Resp(
       role=role,
       text=text,
       thinking=thinking,
       tool_use=tool_use,
       ...
   )
   resp.raw = output
   resp.meta.update(meta)

9. Return resp
```

##### Streaming Extraction Pipeline

**OpenAIChat.from_streamed_result() Algorithm**:
```
STEP 1: Extract deltas from chunk
    text_delta = _extract_text_delta_chat(output)
    tool_deltas = _extract_tool_deltas_chat(output)

STEP 2: Create DeltaResp (FIRST!)
    choice = output.get("choices", [{}])[0]
    delta_resp = DeltaResp(
        text=text_delta,
        finish_reason=choice.get("finish_reason"),
        usage=output.get("usage")
    )

STEP 3: Get staging from prev_resp
    staging = _get_staging(prev_resp, "_staging_chat")

STEP 4: Update staging with tool deltas
    staging = _update_staging_chat(staging, tool_deltas)

STEP 5: Accumulate text
    accumulated_text = _accumulate_text(
        prev_resp.text if prev_resp else None,
        text_delta
    )

STEP 6: Get role
    delta = choice.get("delta", {})
    role = delta.get("role")
    if role is None and prev_resp:
        role = prev_resp.role
    if role is None:
        role = "assistant"

STEP 7: Detect complete tools
    complete_tools, staging = _detect_complete_tools(staging)

STEP 8: Create ToolUse objects for complete tools
    tools = extract_tools_from_messages(messages)
    tool_use = []
    for tool_dict in complete_tools:
        tu = _create_tool_use(tool_dict, tools)
        if tu is not None:
            tool_use.append(tu)

    # Add previous tool_use (already completed)
    if prev_resp and prev_resp.tool_use:
        tool_use = prev_resp.tool_use + tool_use

STEP 9: Construct Resp
    resp = Resp(
        role=role,
        text=accumulated_text,
        tool_use=tool_use,
        finish_reason=choice.get("finish_reason"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )
    resp.raw = output
    _store_staging(resp, "_staging_chat", staging)

STEP 10: Return (resp, delta_resp)
```

**OpenAIResp.from_streamed_result() Algorithm**:
```
STEP 1: Extract deltas from event
    text_delta = _extract_text_delta_resp(output)
    thinking_delta = _extract_thinking_delta_resp(output)
    tool_event = _extract_tool_deltas_resp(output)

STEP 2: Create DeltaResp (FIRST!)
    delta_resp = DeltaResp(
        text=text_delta,
        thinking=thinking_delta,
        finish_reason=output.get("finish_reason"),
        usage=output.get("usage")
    )

STEP 3: Get staging from prev_resp
    staging = _get_staging(prev_resp, "_staging_resp")

STEP 4: Update staging with tool event (if any)
    if tool_event is not None:
        staging = _update_staging_resp(staging, tool_event)

STEP 5: Accumulate text and thinking
    accumulated_text = _accumulate_text(
        prev_resp.text if prev_resp else None,
        text_delta
    )
    accumulated_thinking = _accumulate_text(
        prev_resp.thinking if prev_resp else None,
        thinking_delta
    )

STEP 6: Get role
    role = "assistant"  # Responses API doesn't stream role
    if prev_resp:
        role = prev_resp.role

STEP 7: Detect complete tools
    complete_tools, staging = _detect_complete_tools(staging)

STEP 8: Create ToolUse objects for complete tools
    tools = extract_tools_from_messages(messages)
    tool_use = []
    for tool_dict in complete_tools:
        tu = _create_tool_use(tool_dict, tools)
        if tu is not None:
            tool_use.append(tu)

    # Add previous tool_use (already completed)
    if prev_resp and prev_resp.tool_use:
        tool_use = prev_resp.tool_use + tool_use

STEP 9: Construct Resp
    resp = Resp(
        role=role,
        text=accumulated_text,
        thinking=accumulated_thinking,
        tool_use=tool_use,
        finish_reason=output.get("finish_reason"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )
    resp.raw = output
    _store_staging(resp, "_staging_resp", staging)

STEP 10: Return (resp, delta_resp)
```

##### Helper Function Algorithms

**_extract_tool_calls_chat(message: dict) -> list[dict]**:
```
1. Get tool_calls array
   tool_calls = message.get("tool_calls", [])

2. If empty, return []
   if not tool_calls:
       return []

3. Convert each to canonical format
   result = []
   for tc in tool_calls:
       result.append({
           "id": tc.get("id"),
           "type": tc.get("type", "function"),
           "function": {
               "name": tc.get("function", {}).get("name"),
               "arguments": tc.get("function", {}).get("arguments", "")
           }
       })

4. Return result (already in canonical format, ready for _create_tool_use)
```

**_extract_tool_calls_resp(output: list[dict]) -> list[dict]**:
```
1. Filter for function_call items
   function_calls = [item for item in output if item.get("type") == "function_call"]

2. Convert each to canonical format
   result = []
   for fc in function_calls:
       result.append({
           "id": fc.get("call_id"),  # Map call_id to id
           "type": "function",
           "function": {
               "name": fc.get("name"),
               "arguments": fc.get("arguments", "")
           }
       })

3. Return result (already in canonical format, ready for _create_tool_use)
```

**_create_tool_use(tool_call: dict, available_tools: list[BaseTool]) -> ToolUse | None**:
```
1. Extract fields from canonical format
   tool_id = tool_call.get("id")
   func = tool_call.get("function", {})
   name = func.get("name")
   arguments_str = func.get("arguments", "")

2. Validate name exists
   if not name:
       return None

3. Parse JSON arguments
   try:
       args_dict = json.loads(arguments_str)
   except json.JSONDecodeError:
       log warning "Invalid JSON for tool {name}"
       return None

4. Check args is dict
   if not isinstance(args_dict, dict):
       log warning "Arguments not a dict for tool {name}"
       return None

5. Find BaseTool by name
   tool = None
   for t in available_tools:
       if t.name == name:
           tool = t
           break

   if tool is None:
       log warning "Tool {name} not found in available tools"
       return None

6. Validate with input_model
   try:
       inputs = tool.input_model(**args_dict)
   except ValidationError as e:
       log error "Validation failed for tool {name}: {e}"
       return None

7. Create ToolUse
   tool_use = ToolUse(
       tool_id=tool_id,
       option=tool,
       inputs=inputs
   )

8. Return tool_use
```

**_detect_complete_tools(staging: dict) -> tuple[list[dict], dict]**:
```
1. Initialize result lists
   newly_complete = []
   updated_staging = {}

2. Iterate staging (canonical format)
   for key, tool_call in staging.items():
       # Get function data
       func = tool_call.get("function", {})
       name = func.get("name")
       arguments = func.get("arguments", "")
       was_complete = tool_call.get("_complete", False)

       # Check if NOW complete (has name + valid JSON)
       now_complete = bool(name) and _is_valid_json(arguments)

       if now_complete and not was_complete:
           # Newly completed! Add to result and mark complete
           tool_copy = tool_call.copy()
           tool_copy["_complete"] = True
           newly_complete.append(tool_copy)
           updated_staging[key] = tool_copy
       else:
           # Still incomplete or already was complete
           updated_staging[key] = tool_call

3. Return (newly_complete, updated_staging)
```

**_is_valid_json(s: str) -> bool**:
```
1. Handle empty/None
   if not s:
       return False

2. Try parse
   try:
       json.loads(s)
       return True
   except:
       return False
```

**_accumulate_text(prev: str | None, delta: str | None) -> str**:
```
1. Handle None cases
   if prev is None:
       prev = ""
   if delta is None:
       delta = ""

2. Concatenate
   return prev + delta
```

**_extract_text_delta_chat(chunk: dict) -> str | None**:
```
1. Navigate to delta.content
   choices = chunk.get("choices", [])
   if not choices:
       return None

   delta = choices[0].get("delta", {})
   return delta.get("content")
```

**_extract_text_delta_resp(event: dict) -> str | None**:
```
1. Check event type
   if event.get("type") != "response.text.delta":
       return None

2. Extract delta field
   return event.get("delta")
```

**_extract_thinking_delta_resp(event: dict) -> str | None**:
```
1. Check event type
   if event.get("type") != "response.reasoning.delta":
       return None

2. Extract delta field
   return event.get("delta")
```

**_extract_tool_deltas_chat(chunk: dict) -> list[dict]**:
```
1. Navigate to delta.tool_calls
   choices = chunk.get("choices", [])
   if not choices:
       return []

   delta = choices[0].get("delta", {})
   tool_calls = delta.get("tool_calls", [])

2. Return tool_calls array (already in correct format)
   return tool_calls
```

**_extract_tool_deltas_resp(event: dict) -> dict | None**:
```
1. Check event type and extract data
   event_type = event.get("type")

   if event_type == "response.output_item.added":
       item = event.get("item", {})
       if item.get("type") == "function_call":
           return {
               "action": "add",
               "item_id": item.get("id"),
               "call_id": item.get("call_id"),
               "name": item.get("name"),
               "arguments": item.get("arguments", "")
           }

   elif event_type == "response.function_call_arguments.delta":
       return {
           "action": "append",
           "item_id": event.get("item_id"),
           "delta": event.get("delta", "")
       }

   elif event_type == "response.function_call_arguments.done":
       return {
           "action": "finalize",
           "item_id": event.get("item_id"),
           "arguments": event.get("arguments", "")
       }

2. Return None if not a tool event
   return None
```

**_update_staging_chat(staging: dict, tool_deltas: list[dict]) -> dict**:
```
1. Create new staging (don't mutate)
   new_staging = {k: v.copy() if isinstance(v, dict) else v for k, v in staging.items()}

2. Process each delta
   for delta in tool_deltas:
       idx = delta.get("index")
       if idx is None:
           continue

       # Ensure entry exists in canonical format
       if idx not in new_staging:
           new_staging[idx] = {
               "id": None,
               "type": "function",
               "function": {
                   "name": None,
                   "arguments": ""
               },
               "_complete": False
           }

       # Update fields
       if delta.get("id") is not None:
           new_staging[idx]["id"] = delta["id"]

       if delta.get("type") is not None:
           new_staging[idx]["type"] = delta["type"]

       func = delta.get("function", {})
       if func.get("name") is not None:
           new_staging[idx]["function"]["name"] = func["name"]

       if func.get("arguments") is not None:
           new_staging[idx]["function"]["arguments"] += func["arguments"]

3. Return new_staging
```

**_update_staging_resp(staging: dict, event_data: dict) -> dict**:
```
1. Create new staging (don't mutate)
   new_staging = {k: v.copy() if isinstance(v, dict) else v for k, v in staging.items()}

2. Get action
   action = event_data.get("action")
   item_id = event_data.get("item_id")

3. Handle action
   if action == "add":
       new_staging[item_id] = {
           "id": event_data.get("call_id"),  # Map call_id to id
           "type": "function",
           "function": {
               "name": event_data.get("name"),
               "arguments": event_data.get("arguments", "")
           },
           "_complete": False
       }

   elif action == "append":
       if item_id in new_staging:
           new_staging[item_id]["function"]["arguments"] += event_data.get("delta", "")

   elif action == "finalize":
       if item_id in new_staging:
           new_staging[item_id]["function"]["arguments"] = event_data.get("arguments", "")

4. Return new_staging
```

**_get_staging(resp: Resp | None, key: str) -> dict**:
```
1. Check if resp exists
   if resp is None:
       return {}

2. Get from _raw
   return resp._raw.get(key, {})
```

**_store_staging(resp: Resp, key: str, staging: dict) -> None**:
```
1. Store in _raw
   resp._raw[key] = staging
```

##### Error Path Handling

**For each helper function**:

1. **Missing/None inputs**: Return None or empty value
2. **Invalid JSON**: Log warning, return None (non-streaming) or keep in staging (streaming)
3. **Missing tool**: Log warning, skip, continue with other tools
4. **Validation error**: Log error with details, skip, continue with other tools
5. **Malformed structure**: Try to extract what's possible, log warning
6. **Never crash**: All errors caught and logged, always return something valid

**Example error handling in _create_tool_use**:
```
if not name:
    # Silent fail - incomplete tool call
    return None

try:
    args_dict = json.loads(arguments_str)
except json.JSONDecodeError as e:
    logger.warning(f"Invalid JSON for tool {name}: {e}")
    return None

if tool is None:
    logger.warning(f"Tool '{name}' not found in available tools")
    return None

try:
    inputs = tool.input_model(**args_dict)
except ValidationError as e:
    logger.error(f"Validation failed for tool '{name}': {e}")
    return None
```

##### Edge Cases Handled

1. **Empty content**: Return "" not None
2. **No tool_calls field**: Return [] not error
3. **Tool call with no name**: Skip, don't create ToolUse
4. **Incomplete JSON**: Keep in staging (streaming), skip (non-streaming)
5. **Multiple tool calls**: Process all independently
6. **Tool call spans many chunks**: Accumulate until complete
7. **Tool completed then more deltas**: Ignore (already removed from staging)
8. **First chunk with no prev_resp**: Handle gracefully
9. **Null/None in any field**: Use sensible default
10. **Missing choices array**: Use empty array, extract what's possible

##### Summary

**Pipeline Characteristics**:
- ✅ **Sequential**: Clear step-by-step flow
- ✅ **Simple**: Each step is straightforward
- ✅ **Testable**: Each step can be tested independently
- ✅ **Consistent**: Non-streaming and streaming use same helpers
- ✅ **Robust**: Handles all edge cases without crashing
- ✅ **Clear errors**: Logs specific warnings/errors for debugging

---

### Phase 3: Detailed Implementation Design

#### Subplan 3.1: Helper Functions Design
**Status**: ✅ COMPLETED
**Purpose**: Design each helper function in detail (NOT implement yet)

**Activities**: For each helper function we decide to create:
- Define exact signature with types
- Define input validation
- Define processing algorithm
- Define output format
- Define error cases
- Design unit tests

**Acceptance Criteria**:
- [x] Each function has clear, single purpose
- [x] Signatures are fully typed
- [x] Algorithms are simple and obvious
- [x] Error handling is complete
- [x] Functions are pure/stateless where possible
- [x] Test cases cover all paths

**Findings**:

##### Complete Function Signatures

**Utility Functions**:
```python
def _is_valid_json(s: str) -> bool:
    """Check if string is valid JSON.

    Args:
        s: String to validate

    Returns:
        True if valid JSON, False otherwise

    Edge cases:
        - None/empty string: False
        - Whitespace only: False
        - Invalid JSON: False
    """

def _get_staging(resp: Resp | None, key: str) -> dict[Any, dict]:
    """Get staging dict from prev_resp._raw.

    Args:
        resp: Previous response (or None for first chunk)
        key: Staging key ("_staging_chat" or "_staging_resp")

    Returns:
        Staging dict (empty if resp is None)

    Edge cases:
        - resp is None: Return {}
        - key not in _raw: Return {}
    """

def _store_staging(resp: Resp, key: str, staging: dict[Any, dict]) -> None:
    """Store staging dict in resp._raw.

    Args:
        resp: Response object to update
        key: Staging key
        staging: Staging dict to store

    Side effects:
        Mutates resp._raw[key]
    """

def _accumulate_text(prev: str | None, delta: str | None) -> str:
    """Concatenate previous text with delta.

    Args:
        prev: Previous accumulated text (or None)
        delta: New text fragment (or None)

    Returns:
        Concatenated string (empty if both None)

    Edge cases:
        - Both None: ""
        - prev None: delta or ""
        - delta None: prev or ""
    """
```

**Delta Extraction Functions**:
```python
def _extract_text_delta_chat(chunk: dict[str, Any]) -> str | None:
    """Extract text delta from Chat Completions chunk.

    Args:
        chunk: Raw API chunk

    Returns:
        Text fragment or None if not present

    Edge cases:
        - No choices: None
        - Empty choices: None
        - No delta: None
        - No content in delta: None
    """

def _extract_text_delta_resp(event: dict[str, Any]) -> str | None:
    """Extract text delta from Responses API event.

    Args:
        event: Raw API event

    Returns:
        Text fragment or None if not text delta event

    Edge cases:
        - Wrong event type: None
        - No delta field: None
    """

def _extract_thinking_delta_resp(event: dict[str, Any]) -> str | None:
    """Extract thinking delta from Responses API event.

    Args:
        event: Raw API event

    Returns:
        Thinking fragment or None if not reasoning delta event

    Edge cases:
        - Wrong event type: None
        - No delta field: None
    """

def _extract_tool_deltas_chat(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool call deltas from Chat Completions chunk.

    Args:
        chunk: Raw API chunk

    Returns:
        List of tool call delta dicts (may be empty)

    Edge cases:
        - No choices: []
        - No delta: []
        - No tool_calls in delta: []
    """

def _extract_tool_deltas_resp(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract tool call data from Responses API event.

    Args:
        event: Raw API event

    Returns:
        Dict with action and data, or None if not tool event
        Format: {"action": "add"|"append"|"finalize", "item_id": str, ...}

    Edge cases:
        - Not a tool event: None
        - Missing fields: Use defaults
    """
```

**Non-Streaming Extraction Functions**:
```python
def _extract_tool_calls_chat(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from Chat Completions message (canonical format).

    Args:
        message: Message dict from choices[0].message

    Returns:
        List of tool call dicts in canonical format

    Canonical format:
        {"id": str, "type": str, "function": {"name": str, "arguments": str}}

    Edge cases:
        - No tool_calls: []
        - Empty tool_calls: []
        - Missing fields: Use None/defaults
    """

def _extract_tool_calls_resp(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract tool calls from Responses API output (canonical format).

    Args:
        output: Output array from response.output

    Returns:
        List of tool call dicts in canonical format

    Canonical format:
        {"id": str, "type": str, "function": {"name": str, "arguments": str}}

    Edge cases:
        - No function_call items: []
        - Missing fields: Use None/defaults
    """
```

**Staging Update Functions**:
```python
def _update_staging_chat(
    staging: dict[int, dict[str, Any]],
    tool_deltas: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Update staging with tool deltas from Chat Completions (canonical format).

    Args:
        staging: Current staging dict (keyed by index)
        tool_deltas: Tool call deltas from chunk

    Returns:
        Updated staging dict (new copy, not mutated)

    Staging format (canonical):
        {
            index: {
                "id": str,
                "type": "function",
                "function": {"name": str, "arguments": str},
                "_complete": bool
            }
        }

    Edge cases:
        - Empty tool_deltas: Return staging copy
        - Missing index: Skip delta
        - New tool (index not in staging): Create entry
        - Existing tool: Append to arguments
    """

def _update_staging_resp(
    staging: dict[str, dict[str, Any]],
    event_data: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Update staging with event from Responses API (canonical format).

    Args:
        staging: Current staging dict (keyed by item_id)
        event_data: Event data with action and fields

    Returns:
        Updated staging dict (new copy, not mutated)

    Staging format (canonical):
        {
            item_id: {
                "id": str,  # from call_id
                "type": "function",
                "function": {"name": str, "arguments": str},
                "_complete": bool
            }
        }

    Edge cases:
        - action="add": Create new entry
        - action="append": Append to arguments (skip if item_id not found)
        - action="finalize": Replace arguments (skip if item_id not found)
    """
```

**Tool Completion Detection**:
```python
def _detect_complete_tools(
    staging: dict[Any, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    """Detect newly-complete tools in staging (canonical format).

    Args:
        staging: Current staging dict (canonical format)

    Returns:
        Tuple of (newly_complete_tools, updated_staging)
        - newly_complete_tools: List of tool dicts that just became complete
        - updated_staging: Staging with _complete flags updated

    Completion criteria:
        - function.name exists
        - function.arguments is valid JSON
        - _complete was False (not already complete)

    Edge cases:
        - Empty staging: ([], {})
        - All incomplete: ([], staging)
        - All already complete: ([], staging)
        - Some newly complete: Mark as complete, return them
    """
```

**ToolUse Creation**:
```python
def _create_tool_use(
    tool_call: dict[str, Any],
    available_tools: list[BaseTool]
) -> ToolUse | None:
    """Create ToolUse from tool call dict (canonical format).

    Args:
        tool_call: Tool call dict in canonical format
        available_tools: List of available BaseTool objects

    Returns:
        ToolUse object or None if creation fails

    Failure cases (logs warning/error, returns None):
        - No name: Silent return None
        - Invalid JSON: Log warning, return None
        - Arguments not dict: Log warning, return None
        - Tool not found: Log warning, return None
        - Validation error: Log error with details, return None

    Edge cases:
        - Empty arguments: Parse as empty dict
        - Missing id: ToolUse created with None tool_id
    """
```

##### Test Cases for Each Function

**_is_valid_json**:
```python
def test_is_valid_json_returns_true_for_valid():
    assert _is_valid_json('{"x": 1}') == True
    assert _is_valid_json('[]') == True
    assert _is_valid_json('null') == True

def test_is_valid_json_returns_false_for_invalid():
    assert _is_valid_json('') == False
    assert _is_valid_json(None) == False  # type: ignore
    assert _is_valid_json('{') == False
    assert _is_valid_json('{"x":') == False
```

**_accumulate_text**:
```python
def test_accumulate_text_concatenates():
    assert _accumulate_text("Hello", " world") == "Hello world"

def test_accumulate_text_handles_none():
    assert _accumulate_text(None, "Hi") == "Hi"
    assert _accumulate_text("Hi", None) == "Hi"
    assert _accumulate_text(None, None) == ""
```

**_extract_text_delta_chat**:
```python
def test_extract_text_delta_chat_returns_content():
    chunk = {"choices": [{"delta": {"content": "Hello"}}]}
    assert _extract_text_delta_chat(chunk) == "Hello"

def test_extract_text_delta_chat_returns_none_when_missing():
    assert _extract_text_delta_chat({}) is None
    assert _extract_text_delta_chat({"choices": []}) is None
    assert _extract_text_delta_chat({"choices": [{"delta": {}}]}) is None
```

**_extract_tool_calls_chat**:
```python
def test_extract_tool_calls_chat_returns_canonical_format():
    message = {
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": '{"x":1}'}
        }]
    }
    result = _extract_tool_calls_chat(message)
    assert len(result) == 1
    assert result[0]["id"] == "call_1"
    assert result[0]["function"]["name"] == "foo"

def test_extract_tool_calls_chat_returns_empty_when_none():
    assert _extract_tool_calls_chat({}) == []
    assert _extract_tool_calls_chat({"tool_calls": []}) == []
```

**_update_staging_chat**:
```python
def test_update_staging_chat_creates_new_entry():
    staging = {}
    deltas = [{"index": 0, "id": "call_1", "function": {"name": "foo"}}]
    result = _update_staging_chat(staging, deltas)

    assert 0 in result
    assert result[0]["id"] == "call_1"
    assert result[0]["function"]["name"] == "foo"
    assert result[0]["_complete"] == False

def test_update_staging_chat_appends_arguments():
    staging = {
        0: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": "{\"x\":"},
            "_complete": False
        }
    }
    deltas = [{"index": 0, "function": {"arguments": "1}"}}]
    result = _update_staging_chat(staging, deltas)

    assert result[0]["function"]["arguments"] == '{"x":1}'

def test_update_staging_chat_does_not_mutate_input():
    staging = {}
    deltas = [{"index": 0, "id": "call_1"}]
    result = _update_staging_chat(staging, deltas)

    assert staging == {}
    assert 0 in result
```

**_detect_complete_tools**:
```python
def test_detect_complete_tools_detects_newly_complete():
    staging = {
        0: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": '{"x":1}'},
            "_complete": False
        }
    }
    complete, updated = _detect_complete_tools(staging)

    assert len(complete) == 1
    assert complete[0]["id"] == "call_1"
    assert updated[0]["_complete"] == True

def test_detect_complete_tools_ignores_already_complete():
    staging = {
        0: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": '{"x":1}'},
            "_complete": True
        }
    }
    complete, updated = _detect_complete_tools(staging)

    assert len(complete) == 0
    assert updated[0]["_complete"] == True

def test_detect_complete_tools_keeps_incomplete():
    staging = {
        0: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": '{"x":'},
            "_complete": False
        }
    }
    complete, updated = _detect_complete_tools(staging)

    assert len(complete) == 0
    assert updated[0]["_complete"] == False
```

**_create_tool_use**:
```python
def test_create_tool_use_creates_from_canonical():
    tool = Tool(name="foo", description="Test", input_model=SomeModel)
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "foo", "arguments": '{"x":1}'}
    }

    result = _create_tool_use(tool_call, [tool])

    assert result is not None
    assert result.tool_id == "call_1"
    assert result.option == tool

def test_create_tool_use_returns_none_for_invalid_json():
    tool = Tool(name="foo", description="Test", input_model=SomeModel)
    tool_call = {
        "id": "call_1",
        "function": {"name": "foo", "arguments": '{"x":'}
    }

    result = _create_tool_use(tool_call, [tool])
    assert result is None

def test_create_tool_use_returns_none_for_missing_tool():
    tool = Tool(name="bar", description="Test", input_model=SomeModel)
    tool_call = {
        "id": "call_1",
        "function": {"name": "foo", "arguments": '{"x":1}'}
    }

    result = _create_tool_use(tool_call, [tool])
    assert result is None
```

##### Summary

**All 13+ helper functions have**:
- ✅ Complete type signatures
- ✅ Clear docstrings with edge cases
- ✅ Simple algorithms (already defined in 2.3)
- ✅ Comprehensive test cases
- ✅ Error handling specified

**Next**: Subplan 3.2 - Adapter Methods Design

---

#### Subplan 3.2: Adapter Methods Design
**Status**: ✅ COMPLETED
**Purpose**: Design adapter method implementations in detail

**Activities**: For OpenAIChat and OpenAIResp:
- Design `from_result()` implementation
- Design `from_streamed_result()` implementation
- Define how they call helpers
- Define error handling
- Design integration tests

**Acceptance Criteria**:
- [x] Methods have clear, sequential logic
- [x] Minimal branching/complexity
- [x] Proper delegation to helpers
- [x] Consistent between Chat and Responses adapters
- [x] Error handling is complete
- [x] Test strategy covers all scenarios

**Findings**:

The adapters are simple orchestrators that call helpers in sequence. All complexity is in the helpers (already designed in 3.1).

##### OpenAIChat Methods

**from_result() - Non-Streaming**:
```python
def from_result(self, output: dict, messages: Msg | BaseDialog) -> Resp:
    # Get available tools
    tools = extract_tools_from_messages(messages)

    # Extract components
    choice = output.get("choices", [{}])[0]
    message = choice.get("message", {})

    # Extract tool calls (canonical format)
    tool_call_dicts = _extract_tool_calls_chat(message)

    # Create ToolUse objects
    tool_use = []
    for tc in tool_call_dicts:
        tu = _create_tool_use(tc, tools)
        if tu is not None:
            tool_use.append(tu)

    # Create Resp
    resp = Resp(
        role=message.get("role", "assistant"),
        text=message.get("content") or "",
        tool_use=tool_use,
        finish_reason=choice.get("finish_reason"),
        logprobs=choice.get("logprobs"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )

    # Store raw and metadata
    resp._raw = output
    resp.meta.update(extract_commonly_useful_meta(output))

    return resp
```

**from_streamed_result() - Streaming**:
```python
def from_streamed_result(
    self,
    output: dict,
    messages: Msg | BaseDialog,
    prev_resp: Resp | None
) -> tuple[Resp, DeltaResp]:
    # STEP 1: Extract deltas
    text_delta = _extract_text_delta_chat(output)
    tool_deltas = _extract_tool_deltas_chat(output)

    # STEP 2: Create DeltaResp FIRST
    choice = output.get("choices", [{}])[0]
    delta_resp = DeltaResp(
        text=text_delta,
        finish_reason=choice.get("finish_reason"),
        usage=output.get("usage")
    )

    # STEP 3-4: Update staging
    staging = _get_staging(prev_resp, "_staging_chat")
    staging = _update_staging_chat(staging, tool_deltas)

    # STEP 5: Accumulate text
    accumulated_text = _accumulate_text(
        prev_resp.text if prev_resp else None,
        text_delta
    )

    # STEP 6: Get role
    delta = choice.get("delta", {})
    role = delta.get("role")
    if role is None and prev_resp:
        role = prev_resp.role
    if role is None:
        role = "assistant"

    # STEP 7-8: Detect and create complete tools
    complete_tools, staging = _detect_complete_tools(staging)
    tools = extract_tools_from_messages(messages)
    tool_use = []
    for tc in complete_tools:
        tu = _create_tool_use(tc, tools)
        if tu is not None:
            tool_use.append(tu)

    # Add previous tool_use
    if prev_resp and prev_resp.tool_use:
        tool_use = prev_resp.tool_use + tool_use

    # STEP 9: Construct Resp
    resp = Resp(
        role=role,
        text=accumulated_text,
        tool_use=tool_use,
        finish_reason=choice.get("finish_reason"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )
    resp._raw = output
    _store_staging(resp, "_staging_chat", staging)

    # STEP 10: Return
    return (resp, delta_resp)
```

##### OpenAIResp Methods

**from_result() - Non-Streaming**:
```python
def from_result(self, output: dict, messages: Msg | BaseDialog) -> Resp:
    # Get available tools
    tools = extract_tools_from_messages(messages)

    # Extract output items
    output_items = output.get("output", [])

    # Find message item
    message_item = None
    for item in output_items:
        if item.get("type") == "message":
            message_item = item
            break

    if message_item is None:
        message_item = {}

    # Extract tool calls (canonical format)
    tool_call_dicts = _extract_tool_calls_resp(output_items)

    # Create ToolUse objects
    tool_use = []
    for tc in tool_call_dicts:
        tu = _create_tool_use(tc, tools)
        if tu is not None:
            tool_use.append(tu)

    # Extract other fields
    choice = output.get("choices", [{}])[0]

    # Create Resp
    resp = Resp(
        role=message_item.get("role", "assistant"),
        text=message_item.get("content") or "",
        thinking=output.get("reasoning"),
        tool_use=tool_use,
        finish_reason=choice.get("finish_reason"),
        logprobs=choice.get("logprobs"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )

    # Store raw and metadata
    resp._raw = output
    resp.meta.update(extract_commonly_useful_meta(output))

    return resp
```

**from_streamed_result() - Streaming**:
```python
def from_streamed_result(
    self,
    output: dict,
    messages: Msg | BaseDialog,
    prev_resp: Resp | None
) -> tuple[Resp, DeltaResp]:
    # STEP 1: Extract deltas
    text_delta = _extract_text_delta_resp(output)
    thinking_delta = _extract_thinking_delta_resp(output)
    tool_event = _extract_tool_deltas_resp(output)

    # STEP 2: Create DeltaResp FIRST
    delta_resp = DeltaResp(
        text=text_delta,
        thinking=thinking_delta,
        finish_reason=output.get("finish_reason"),
        usage=output.get("usage")
    )

    # STEP 3-4: Update staging
    staging = _get_staging(prev_resp, "_staging_resp")
    if tool_event is not None:
        staging = _update_staging_resp(staging, tool_event)

    # STEP 5: Accumulate text and thinking
    accumulated_text = _accumulate_text(
        prev_resp.text if prev_resp else None,
        text_delta
    )
    accumulated_thinking = _accumulate_text(
        prev_resp.thinking if prev_resp else None,
        thinking_delta
    )

    # STEP 6: Get role
    role = "assistant"
    if prev_resp:
        role = prev_resp.role

    # STEP 7-8: Detect and create complete tools
    complete_tools, staging = _detect_complete_tools(staging)
    tools = extract_tools_from_messages(messages)
    tool_use = []
    for tc in complete_tools:
        tu = _create_tool_use(tc, tools)
        if tu is not None:
            tool_use.append(tu)

    # Add previous tool_use
    if prev_resp and prev_resp.tool_use:
        tool_use = prev_resp.tool_use + tool_use

    # STEP 9: Construct Resp
    resp = Resp(
        role=role,
        text=accumulated_text,
        thinking=accumulated_thinking,
        tool_use=tool_use,
        finish_reason=output.get("finish_reason"),
        id=output.get("id"),
        model=output.get("model"),
        usage=output.get("usage") or {}
    )
    resp._raw = output
    _store_staging(resp, "_staging_resp", staging)

    # STEP 10: Return
    return (resp, delta_resp)
```

##### Integration Test Strategy

**Non-Streaming Tests**:
```python
def test_openai_chat_from_result_extracts_all_fields():
    # Full Chat Completions response with tool calls
    output = {...}
    messages = Prompt(text="test", tools=[weather_tool])

    resp = adapter.from_result(output, messages)

    assert resp.role == "assistant"
    assert resp.text == "The weather is..."
    assert len(resp.tool_use) == 1
    assert resp.tool_use[0].option.name == "get_weather"
    assert resp.finish_reason == "tool_calls"
    assert resp.model == "gpt-4"

def test_openai_resp_from_result_extracts_thinking():
    # Full Responses API response with reasoning
    output = {...}
    messages = Prompt(text="test")

    resp = adapter.from_result(output, messages)

    assert resp.thinking == "Let me think..."
    assert resp.text == "The answer is..."
```

**Streaming Tests**:
```python
def test_openai_chat_streaming_accumulates_text():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]

    prev = None
    for chunk in chunks:
        resp, delta = adapter.from_streamed_result(chunk, messages, prev)
        prev = resp

    assert prev.text == "Hello world"

def test_openai_chat_streaming_builds_tool_calls():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "foo"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]}}]},
    ]

    prev = None
    for chunk in chunks:
        resp, delta = adapter.from_streamed_result(chunk, messages, prev)
        prev = resp

    assert len(prev.tool_use) == 1
    assert prev.tool_use[0].option.name == "foo"

def test_openai_resp_streaming_accumulates_thinking():
    events = [
        {"type": "response.reasoning.delta", "delta": "Let "},
        {"type": "response.reasoning.delta", "delta": "me think"},
    ]

    prev = None
    for event in events:
        resp, delta = adapter.from_streamed_result(event, messages, prev)
        prev = resp

    assert prev.thinking == "Let me think"
```

##### Summary

**Adapter Characteristics**:
- ✅ **Simple orchestration**: Just call helpers in sequence
- ✅ **Minimal logic**: All complexity in helpers
- ✅ **Consistent structure**: Both adapters follow same pattern
- ✅ **Clear steps**: 10 steps for streaming, ~8 for non-streaming
- ✅ **No branching**: Linear flow through helpers
- ✅ **Easy to test**: Mock helpers or use real responses

**Phase 3 Complete**: All functions and methods fully designed with signatures, algorithms, and tests!

---

### Phase 4: Test Strategy

#### Subplan 4.1: Test Plan Design
**Status**: ✅ COMPLETED
**Purpose**: Comprehensive testing strategy

**Activities**:
- Design unit tests for helpers
- Design integration tests for adapters
- Design end-to-end tests
- Identify test data needs
- Design validation approach

**Acceptance Criteria**:
- [x] Every requirement has corresponding tests
- [x] Edge cases are covered
- [x] Error paths are tested
- [x] Streaming and non-streaming both tested
- [x] Both APIs tested equivalently
- [x] Tests are simple and maintainable

**Findings**:

##### Test Organization

**Test Files**:
```
tests/proc/
  test_openai_helpers.py      # Unit tests for ~13 helper functions
  test_openai_adapters.py     # Integration tests for adapters
  test_openai_e2e.py          # End-to-end streaming scenarios
```

**Test Data Location**:
```
tests/proc/fixtures/
  chat_responses.py           # Sample Chat Completions responses
  responses_api_events.py     # Sample Responses API events
  tool_fixtures.py            # Sample tools and ToolUse objects
```

##### Unit Tests (Helper Functions)

Each helper function gets its own test class with comprehensive coverage:

**Test Class Template**:
```python
class Test<FunctionName>:
    """Unit tests for _<function_name> helper."""

    def test_<function>_<behavior>_<condition>(self):
        """Test normal operation."""
        # Test happy path

    def test_<function>_returns_none_when_missing(self):
        """Test missing data."""
        # Test None/empty cases

    def test_<function>_handles_invalid_structure(self):
        """Test malformed input."""
        # Test edge cases

    def test_<function>_does_not_mutate_input(self):
        """Test immutability (for update functions)."""
        # Verify no side effects
```

**Coverage Requirements**:
- ✅ Happy path (valid input → expected output)
- ✅ Empty/None inputs
- ✅ Malformed structure
- ✅ Invalid data types
- ✅ Edge cases (empty strings, whitespace, etc.)
- ✅ Immutability (functions don't mutate inputs)

**Acceptance Tests for Each Helper**:

1. **_is_valid_json**:
   - [ ] Returns True for valid JSON objects
   - [ ] Returns True for valid JSON arrays
   - [ ] Returns False for empty string
   - [ ] Returns False for None
   - [ ] Returns False for incomplete JSON
   - [ ] Returns False for non-JSON strings

2. **_accumulate_text**:
   - [ ] Concatenates two strings correctly
   - [ ] Handles None prev (returns delta)
   - [ ] Handles None delta (returns prev)
   - [ ] Returns empty string for both None
   - [ ] Preserves whitespace

3. **_extract_text_delta_chat**:
   - [ ] Extracts content from valid chunk
   - [ ] Returns None for empty chunk
   - [ ] Returns None for no choices
   - [ ] Returns None for no delta
   - [ ] Returns None for no content in delta

4. **_extract_text_delta_resp**:
   - [ ] Extracts delta from text event
   - [ ] Returns None for wrong event type
   - [ ] Returns None for no delta field

5. **_extract_thinking_delta_resp**:
   - [ ] Extracts delta from reasoning event
   - [ ] Returns None for wrong event type
   - [ ] Returns None for no delta field

6. **_extract_tool_deltas_chat**:
   - [ ] Returns tool_calls array from delta
   - [ ] Returns empty list for no choices
   - [ ] Returns empty list for no tool_calls
   - [ ] Preserves all delta fields

7. **_extract_tool_deltas_resp**:
   - [ ] Returns action dict for output_item.added
   - [ ] Returns action dict for arguments.delta
   - [ ] Returns action dict for arguments.done
   - [ ] Returns None for non-tool events
   - [ ] Extracts all required fields

8. **_extract_tool_calls_chat**:
   - [ ] Returns canonical format for tool_calls
   - [ ] Returns empty list for no tool_calls
   - [ ] Maps fields correctly (id, type, function)
   - [ ] Handles missing function fields gracefully

9. **_extract_tool_calls_resp**:
   - [ ] Returns canonical format for function_call items
   - [ ] Returns empty list for no function_calls
   - [ ] Maps call_id to id correctly
   - [ ] Filters only function_call type items

10. **_update_staging_chat**:
    - [ ] Creates new entry for new index
    - [ ] Updates existing entry
    - [ ] Appends to arguments correctly
    - [ ] Sets id and name on first delta
    - [ ] Returns new dict (doesn't mutate input)
    - [ ] Handles missing index gracefully
    - [ ] Initializes with canonical format

11. **_update_staging_resp**:
    - [ ] Creates entry for "add" action
    - [ ] Appends for "append" action
    - [ ] Replaces for "finalize" action
    - [ ] Skips if item_id not found (append/finalize)
    - [ ] Returns new dict (doesn't mutate input)
    - [ ] Maps call_id to id correctly

12. **_detect_complete_tools**:
    - [ ] Detects newly complete tool (valid JSON + name)
    - [ ] Marks tool as complete (_complete=True)
    - [ ] Ignores already complete tools
    - [ ] Keeps incomplete tools in staging
    - [ ] Returns empty lists for empty staging
    - [ ] Returns updated staging with flags set

13. **_create_tool_use**:
    - [ ] Creates ToolUse from canonical format
    - [ ] Sets tool_id, option, inputs correctly
    - [ ] Returns None for missing name
    - [ ] Returns None for invalid JSON
    - [ ] Returns None for non-dict arguments
    - [ ] Returns None for tool not found
    - [ ] Returns None for validation error
    - [ ] Logs appropriate warnings/errors

14. **_get_staging**:
    - [ ] Returns staging dict from prev_resp._raw
    - [ ] Returns empty dict for None resp
    - [ ] Returns empty dict for missing key

15. **_store_staging**:
    - [ ] Stores staging in resp._raw[key]
    - [ ] Mutates resp._raw as expected

##### Integration Tests (Adapters)

Test each adapter method with realistic API responses:

**OpenAIChat Adapter Tests**:

```python
class TestOpenAIChatFromResult:
    """Integration tests for OpenAIChat.from_result()."""

    def test_from_result_extracts_all_fields(self):
        """AC: Extract role, text, tool_use, finish_reason, id, model, usage."""
        # Full response with all fields
        # Verify all fields populated correctly

    def test_from_result_creates_tool_use_objects(self):
        """AC: Tool calls converted to ToolUse with correct option and inputs."""
        # Response with tool calls
        # Verify ToolUse created, validated, matched to tools

    def test_from_result_handles_no_tool_calls(self):
        """AC: Works without tool_calls field."""
        # Response without tool_calls
        # Verify tool_use is empty list

    def test_from_result_skips_invalid_tool_calls(self):
        """AC: Invalid tool calls logged and skipped."""
        # Response with invalid tool (bad JSON, missing tool, etc.)
        # Verify only valid tools in tool_use

class TestOpenAIChatFromStreamedResult:
    """Integration tests for OpenAIChat.from_streamed_result()."""

    def test_streaming_accumulates_text_across_chunks(self):
        """AC: Text accumulates correctly over multiple chunks."""
        # Multiple chunks with text deltas
        # Verify final text is complete

    def test_streaming_builds_tool_calls(self):
        """AC: Tool calls reconstructed from deltas."""
        # Chunks with tool call deltas (id, name, argument fragments)
        # Verify ToolUse created when complete

    def test_streaming_detects_completion_at_json_boundary(self):
        """AC: Tool marked complete when JSON becomes valid."""
        # Chunks that complete JSON gradually
        # Verify ToolUse appears at right chunk

    def test_streaming_preserves_role_from_first_chunk(self):
        """AC: Role from first chunk used in subsequent chunks."""
        # First chunk with role, subsequent without
        # Verify role preserved

    def test_streaming_returns_delta_and_resp(self):
        """AC: Returns (Resp, DeltaResp) tuple."""
        # Any chunk
        # Verify tuple structure, DeltaResp has only deltas

    def test_streaming_handles_first_chunk_no_prev(self):
        """AC: Works when prev_resp is None."""
        # First chunk with prev_resp=None
        # Verify no errors, staging initialized

    def test_streaming_accumulates_previous_tool_use(self):
        """AC: Previous tool_use preserved and extended."""
        # Chunk completing second tool call
        # Verify both tool calls in tool_use
```

**OpenAIResp Adapter Tests**:

```python
class TestOpenAIRespFromResult:
    """Integration tests for OpenAIResp.from_result()."""

    def test_from_result_extracts_thinking(self):
        """AC: Thinking field extracted from reasoning."""
        # Response with reasoning
        # Verify thinking populated

    def test_from_result_finds_message_item(self):
        """AC: Extracts text from output item with type=message."""
        # Response with multiple output items
        # Verify correct message extracted

    def test_from_result_extracts_function_calls(self):
        """AC: Function calls from output converted to ToolUse."""
        # Response with function_call items
        # Verify ToolUse created correctly

class TestOpenAIRespFromStreamedResult:
    """Integration tests for OpenAIResp.from_streamed_result()."""

    def test_streaming_accumulates_thinking(self):
        """AC: Thinking accumulates from reasoning.delta events."""
        # Multiple reasoning.delta events
        # Verify thinking accumulated

    def test_streaming_handles_multiple_event_types(self):
        """AC: Processes text, reasoning, and tool events."""
        # Mix of event types
        # Verify all processed correctly

    def test_streaming_builds_tools_from_events(self):
        """AC: Tool calls from output_item.added + arguments.delta."""
        # Events: added, delta, delta, done
        # Verify ToolUse created

    def test_streaming_uses_finalize_event(self):
        """AC: arguments.done replaces accumulated arguments."""
        # Events with .done providing final arguments
        # Verify final arguments used
```

##### End-to-End Tests

Test complete streaming scenarios with realistic multi-chunk flows:

**E2E Test Scenarios**:

```python
class TestE2EChatStreaming:
    """End-to-end streaming scenarios for Chat Completions."""

    def test_e2e_text_only_stream(self):
        """AC: Stream text-only response successfully."""
        # 5+ chunks with text deltas
        # Verify complete text, correct DeltaResp at each step

    def test_e2e_tool_call_stream(self):
        """AC: Stream tool call successfully."""
        # Chunks: id+name, arguments fragments, completion
        # Verify ToolUse appears when complete

    def test_e2e_multiple_tool_calls_stream(self):
        """AC: Stream multiple concurrent tool calls."""
        # Interleaved deltas for 2+ tools
        # Verify all tools built correctly

    def test_e2e_text_and_tool_stream(self):
        """AC: Stream text and tool call together."""
        # Mixed text and tool deltas
        # Verify both accumulated correctly

class TestE2EResponsesStreaming:
    """End-to-end streaming scenarios for Responses API."""

    def test_e2e_thinking_and_text_stream(self):
        """AC: Stream reasoning and text."""
        # Reasoning.delta + text.delta events
        # Verify both accumulated separately

    def test_e2e_tool_with_reasoning_stream(self):
        """AC: Stream tool call with reasoning."""
        # Mixed reasoning and tool events
        # Verify both processed correctly
```

##### Test Data Strategy

**Fixture Organization**:

```python
# tests/proc/fixtures/chat_responses.py
CHAT_SIMPLE_TEXT = {
    "id": "chatcmpl-123",
    "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
    ...
}

CHAT_WITH_TOOL_CALLS = {
    "id": "chatcmpl-124",
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "foo", "arguments": '{"x":1}'}}
            ]
        }
    }],
    ...
}

CHAT_STREAMING_CHUNKS = [
    {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
    {"choices": [{"delta": {"content": " world"}}]},
    ...
]

CHAT_STREAMING_TOOL_CHUNKS = [
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "foo"}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]}}]},
]

# tests/proc/fixtures/tool_fixtures.py
@tool
def sample_tool(x: int) -> str:
    return f"Result: {x}"

SAMPLE_TOOLS = [sample_tool]
```

##### Acceptance Criteria Summary

**For Each Implementation Step** (tracked in Phase 5):

After implementing each component, verify:

1. **Helper Functions**:
   - [ ] All unit tests pass
   - [ ] No mutations of input data
   - [ ] Proper error logging
   - [ ] Type hints validated

2. **Adapter Methods**:
   - [ ] All integration tests pass
   - [ ] Both APIs work equivalently
   - [ ] Streaming and non-streaming both work
   - [ ] Tool calls extracted and created correctly

3. **Overall System**:
   - [ ] All E2E tests pass
   - [ ] No regressions in existing tests
   - [ ] Code coverage > 90%
   - [ ] All requirements from Phase 1 met
   - [ ] All invariants maintained

##### Test Execution Strategy

**Run Tests in Order**:
```bash
# 1. Unit tests (fast, isolated)
pytest tests/proc/test_openai_helpers.py -v

# 2. Integration tests (adapter methods)
pytest tests/proc/test_openai_adapters.py -v

# 3. E2E tests (full scenarios)
pytest tests/proc/test_openai_e2e.py -v

# 4. All tests
pytest tests/proc/ -v

# 5. With coverage
pytest tests/proc/ --cov=dachi.proc.openai --cov-report=html
```

**Continuous Validation**:
- Run unit tests after each helper implementation
- Run integration tests after adapter method completion
- Run full suite before committing
- Check coverage to ensure >90%

##### Summary

**Test Coverage**:
- ✅ ~50+ unit tests for helpers
- ✅ ~20+ integration tests for adapters
- ✅ ~10+ E2E tests for streaming scenarios
- ✅ Both APIs tested equivalently
- ✅ All edge cases covered
- ✅ Clear acceptance criteria for each step

**Next**: Phase 5 - Build & Validation Order

---

### Phase 5: Implementation Sequence

#### Subplan 5.1: Build & Validation Order
**Status**: ✅ COMPLETED
**Purpose**: Define implementation sequence with verification at each step

**Activities**:
- Define build order (helpers → adapters → tests)
- Define validation checkpoints
- Define acceptance criteria for each implementation step
- Define regression prevention strategy

**Acceptance Criteria**:
- [x] Clear sequential order
- [x] Each step is independently verifiable
- [x] Rollback strategy if issues found
- [x] Integration points are clear
- [x] Final validation is comprehensive

**Findings**:

##### Implementation Order

**STEP 1: Prepare Environment**
```bash
# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate dachi

# Verify current tests pass
pytest tests/proc/test_process.py -v
```

**Validation**:
- [ ] Environment activated
- [ ] Existing tests baseline established
- [ ] No broken tests to start

---

**STEP 2: Implement Utility Helper Functions**

Implement in this order (simplest → most complex):

1. `_is_valid_json(s: str) -> bool`
2. `_accumulate_text(prev, delta) -> str`
3. `_get_staging(resp, key) -> dict`
4. `_store_staging(resp, key, staging) -> None`

**File**: `dachi/proc/openai.py`

**Validation After Each Function**:
```bash
pytest tests/proc/test_openai_helpers.py::Test<FunctionName> -v
```

**Acceptance Criteria**:
- [ ] All 4 utility functions implemented
- [ ] All unit tests pass
- [ ] No type errors (run `pyright dachi/proc/openai.py`)
- [ ] Functions are pure (no side effects except _store_staging)

---

**STEP 3: Implement Delta Extraction Functions**

Implement in this order:

1. `_extract_text_delta_chat(chunk) -> str | None`
2. `_extract_text_delta_resp(event) -> str | None`
3. `_extract_thinking_delta_resp(event) -> str | None`
4. `_extract_tool_deltas_chat(chunk) -> list[dict]`
5. `_extract_tool_deltas_resp(event) -> dict | None`

**File**: `dachi/proc/openai.py`

**Validation After Each Function**:
```bash
pytest tests/proc/test_openai_helpers.py::Test<FunctionName> -v
```

**Acceptance Criteria**:
- [ ] All 5 extraction functions implemented
- [ ] All unit tests pass
- [ ] Handle missing/None fields gracefully
- [ ] No crashes on malformed input

---

**STEP 4: Implement Non-Streaming Extraction Functions**

Implement in this order:

1. `_extract_tool_calls_chat(message) -> list[dict]`
2. `_extract_tool_calls_resp(output) -> list[dict]`

**File**: `dachi/proc/openai.py`

**Validation After Each Function**:
```bash
pytest tests/proc/test_openai_helpers.py::Test<FunctionName> -v
```

**Acceptance Criteria**:
- [ ] Both functions implemented
- [ ] Return canonical format
- [ ] All unit tests pass
- [ ] Handle empty/missing tool_calls

---

**STEP 5: Implement Staging Update Functions**

Implement in this order:

1. `_update_staging_chat(staging, tool_deltas) -> dict`
2. `_update_staging_resp(staging, event_data) -> dict`

**File**: `dachi/proc/openai.py`

**Validation After Each Function**:
```bash
pytest tests/proc/test_openai_helpers.py::Test<FunctionName> -v
```

**Acceptance Criteria**:
- [ ] Both functions implemented
- [ ] Create canonical format entries
- [ ] Don't mutate input staging
- [ ] All unit tests pass
- [ ] Handle missing fields gracefully

---

**STEP 6: Implement Tool Completion Detection**

Implement:

1. `_detect_complete_tools(staging) -> tuple[list[dict], dict]`

**File**: `dachi/proc/openai.py`

**Validation**:
```bash
pytest tests/proc/test_openai_helpers.py::TestDetectCompleteTools -v
```

**Acceptance Criteria**:
- [ ] Function implemented
- [ ] Detects newly complete (not already complete)
- [ ] Marks tools as complete (_complete=True)
- [ ] All unit tests pass
- [ ] Works with both API staging formats

---

**STEP 7: Implement ToolUse Creation**

Implement:

1. `_create_tool_use(tool_call, available_tools) -> ToolUse | None`

**File**: `dachi/proc/openai.py`

**Validation**:
```bash
pytest tests/proc/test_openai_helpers.py::TestCreateToolUse -v
```

**Acceptance Criteria**:
- [ ] Function implemented
- [ ] Extracts from canonical format
- [ ] Looks up tools correctly
- [ ] Validates with input_model
- [ ] Logs warnings/errors appropriately
- [ ] All unit tests pass

**Checkpoint: All Helpers Complete**:
```bash
pytest tests/proc/test_openai_helpers.py -v
```
- [ ] All ~50 unit tests pass
- [ ] No type errors
- [ ] Code coverage > 95% for helpers

---

**STEP 8: Implement OpenAIChat.from_result()**

Update the existing method in `OpenAIChat` class.

**File**: `dachi/proc/openai.py`

**Implementation**:
- Follow algorithm from Subplan 3.2
- Call helpers in sequence
- No complex logic in adapter

**Validation**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIChatFromResult -v
```

**Acceptance Criteria**:
- [ ] Method implemented
- [ ] Extracts all fields correctly
- [ ] Creates ToolUse objects
- [ ] All integration tests pass
- [ ] Handles missing fields gracefully

---

**STEP 9: Implement OpenAIChat.from_streamed_result()**

Update the existing method in `OpenAIChat` class.

**File**: `dachi/proc/openai.py`

**Implementation**:
- Follow 10-step algorithm from Subplan 3.2
- Create DeltaResp FIRST
- Call helpers in sequence

**Validation**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIChatFromStreamedResult -v
```

**Acceptance Criteria**:
- [ ] Method implemented
- [ ] Returns (Resp, DeltaResp) tuple
- [ ] DeltaResp contains only deltas
- [ ] Resp contains accumulated values
- [ ] Tool calls built correctly
- [ ] All integration tests pass

**Checkpoint: OpenAIChat Complete**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIChat -v
pytest tests/proc/test_openai_e2e.py::TestE2EChatStreaming -v
```
- [ ] All OpenAIChat tests pass
- [ ] E2E streaming tests pass

---

**STEP 10: Implement OpenAIResp.from_result()**

Update the existing method in `OpenAIResp` class.

**File**: `dachi/proc/openai.py`

**Implementation**:
- Follow algorithm from Subplan 3.2
- Extract from output[] array
- Extract thinking field

**Validation**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIRespFromResult -v
```

**Acceptance Criteria**:
- [ ] Method implemented
- [ ] Finds message item correctly
- [ ] Extracts thinking
- [ ] Creates ToolUse objects
- [ ] All integration tests pass

---

**STEP 11: Implement OpenAIResp.from_streamed_result()**

Update the existing method in `OpenAIResp` class.

**File**: `dachi/proc/openai.py`

**Implementation**:
- Follow 10-step algorithm from Subplan 3.2
- Handle multiple event types
- Accumulate thinking separately from text

**Validation**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIRespFromStreamedResult -v
```

**Acceptance Criteria**:
- [ ] Method implemented
- [ ] Handles all event types
- [ ] Accumulates thinking correctly
- [ ] Tool calls built correctly
- [ ] All integration tests pass

**Checkpoint: OpenAIResp Complete**:
```bash
pytest tests/proc/test_openai_adapters.py::TestOpenAIResp -v
pytest tests/proc/test_openai_e2e.py::TestE2EResponsesStreaming -v
```
- [ ] All OpenAIResp tests pass
- [ ] E2E streaming tests pass

---

**STEP 12: Remove Dead Code**

Clean up `dachi/proc/openai.py`:

1. Remove `extract_openai_tool_calls()` stub (lines 164-168)
2. Remove commented code in `OpenAIResp.from_streamed_result()` (lines 551-618)
3. Verify no duplicate code

**Validation**:
```bash
pytest tests/proc/ -v
```

**Acceptance Criteria**:
- [ ] All dead code removed
- [ ] No commented-out code remains
- [ ] All tests still pass
- [ ] Code is clean and readable

---

**STEP 13: Update tests/core/test_msg.py**

Fix broken tests that use old `Resp(msg=...)` constructor.

**File**: `tests/core/test_msg.py`

**Changes**:
```python
# OLD (broken):
def _new_resp(self):
    return Resp(msg=_sample_msg())

# NEW (correct):
def _new_resp(self):
    return Resp(role="user", text="hello")
```

**Validation**:
```bash
pytest tests/core/test_msg.py -v
```

**Acceptance Criteria**:
- [ ] All test_msg.py tests pass
- [ ] No use of `msg=` parameter
- [ ] Tests validate Resp behavior correctly

---

**STEP 14: Final Validation**

Run complete test suite and verify all requirements:

```bash
# All tests
pytest tests/proc/ tests/core/ -v

# With coverage
pytest tests/proc/ --cov=dachi.proc.openai --cov-report=html --cov-report=term

# Type checking
pyright dachi/proc/openai.py
```

**Final Acceptance Criteria**:

**Functional Requirements** (from Phase 1):
- [ ] FR1: Non-streaming Chat Completions extraction works
- [ ] FR2: Non-streaming Responses API extraction works
- [ ] FR3: Streaming Chat Completions extraction works
- [ ] FR4: Streaming Responses API extraction works
- [ ] FR5: Tool call to ToolUse conversion works

**Data Storage Requirements**:
- [ ] DSR1: Resp contains complete values (non-streaming)
- [ ] DSR2: Resp contains accumulated values (streaming)
- [ ] DSR3: DeltaResp contains only deltas (streaming)
- [ ] DSR4: ToolUse objects properly constructed

**Invariants**:
- [ ] INV1: tool_use vs tool_calls distinction maintained
- [ ] INV2: Delta vs accumulated separation maintained
- [ ] INV3: ToolUse only created when complete
- [ ] INV4: DeltaResp created before Resp (streaming)
- [ ] INV5: Text accumulation works correctly

**Test Coverage**:
- [ ] All unit tests pass (~50 tests)
- [ ] All integration tests pass (~20 tests)
- [ ] All E2E tests pass (~10 tests)
- [ ] Code coverage > 90%
- [ ] No test failures
- [ ] No type errors

**Code Quality**:
- [ ] No dead code
- [ ] No commented code
- [ ] Clear, simple functions
- [ ] Proper error logging
- [ ] Type hints throughout

---

##### Rollback Strategy

If any step fails validation:

1. **Identify failing tests**: Run specific test to isolate issue
2. **Check algorithm**: Verify implementation matches design in Phases 2-3
3. **Fix implementation**: Correct code based on design
4. **Re-validate**: Run tests again
5. **If still failing**: Review Phase 2-3 design, may need adjustment

**Nuclear Option**:
```bash
git checkout dachi/proc/openai.py
# Start over from last good commit
```

##### Implementation Notes

**Estimated Time**:
- Steps 1-7 (Helpers): 2-3 hours
- Steps 8-11 (Adapters): 2-3 hours
- Steps 12-14 (Cleanup & Validation): 1 hour
- **Total**: 5-7 hours

**Order is Critical**:
- Must implement helpers before adapters (adapters depend on helpers)
- Must implement utilities before complex helpers (dependencies)
- Must test each function before moving to next (catch bugs early)

**Success Indicators**:
- Each test passes on first try (design is good)
- No surprises during implementation (plan is complete)
- Code is simple and obvious (design is clean)

---

##### Summary

**Implementation Path**:
1. ✅ 4 utility functions
2. ✅ 5 delta extraction functions
3. ✅ 2 non-streaming extraction functions
4. ✅ 2 staging update functions
5. ✅ 1 completion detection function
6. ✅ 1 ToolUse creation function
7. ✅ 4 adapter methods (2 per adapter)
8. ✅ Clean up dead code
9. ✅ Fix broken tests
10. ✅ Final validation

**Total**: ~15 functions + 4 methods + cleanup = **19 implementation tasks**

**All phases complete! Ready to implement.**

---

## Execution Log

### 2025-11-30: Document Created
- Created initial planning document structure
- Ready to begin Subplan 1.1: Current State Audit

### 2025-11-30: Phase 1 Complete - Analysis & Requirements
- ✅ Subplan 1.1: Current State Audit - Documented all message fields, found incomplete code, identified what works/breaks
- ✅ Subplan 1.2: API Contract Analysis - Exhaustive mapping of both APIs (non-streaming + streaming)
- ✅ Subplan 1.3: Requirements Definition - Defined functional requirements, invariants, error handling, testing strategy
- **Key Insights**:
  - DeltaResp must be created FIRST in streaming
  - Tool call completion requires both delta and accumulated state
  - Staging must be stored in prev_resp._raw
  - Both APIs require JSON validation for tool completion

### 2025-11-30: Phase 2 Complete - Architecture Design
- ✅ Subplan 2.1: Data Structure Design - Identified 5 clear processing boundaries with simple responsibilities
- ✅ Subplan 2.2: Module Boundary Design - Defined ~13 helper functions + adapter orchestration, zero circular dependencies
- ✅ Subplan 2.3: Extraction Pipeline Design - Complete step-by-step algorithms for all functions (3-8 steps each)
- **Key Decisions**:
  - All code stays in `dachi/proc/openai.py` (no new modules)
  - Helpers are pure functions (no state, no side effects except logging)
  - Each helper does ONE thing (extract, update, accumulate, detect, create)
  - Adapters just orchestrate (call helpers in sequence)
  - Error handling: log and continue, never crash

### 2025-11-30: API Documentation Deep Review
**Reviewing user-provided reconstruction code for accuracy**

**Critical Details from User's Code Example**:

1. **Canonical Tool Call Format** (both APIs should normalize to this):
```python
{
    "id": "call_id",           # Maps from: Chat.id or Responses.call_id
    "type": "function",
    "function": {
        "name": "tool_name",
        "arguments": "{...}"   # JSON string
    },
    "_complete": False         # Internal tracking
}
```

2. **Chat Completions Streaming Details**:
   - Tool calls arrive in `choices[0].delta.tool_calls[]`
   - Each delta has `index` (position in array)
   - First chunk for a tool has: `id`, `type`, `function.name`
   - Subsequent chunks have: `index`, `function.arguments` (fragments)
   - Must track by index, accumulate arguments
   - Detect completion when: name exists + arguments is valid JSON
   - Track `_complete` flag to detect newly-completed tools

3. **Responses API Streaming Details**:
   - Event: `response.output_item.added` (type=function_call)
     - Provides: `item.id` (item_id), `item.call_id`, `item.name`, `item.arguments` (may be empty)
   - Event: `response.function_call_arguments.delta`
     - Provides: `item_id`, `delta` (fragment to append)
   - Event: `response.function_call_arguments.done`
     - Provides: `item_id`, `arguments` (complete string)
   - Event: `response.output_item.done`
     - Complete item with all fields
   - Must track by `item_id`, accumulate arguments
   - Detect completion on `.done` event OR when arguments becomes valid JSON

4. **Critical Staging Requirements**:
   - Staging must track BOTH APIs in same format
   - Need `_complete` flag to detect "newly completed" vs "already complete"
   - Need to handle case where tool gets more deltas after already complete (ignore them)
   - Staging structure should look like:
```python
# Chat Completions
{
    0: {
        "id": "call_123",
        "type": "function",
        "function": {"name": "foo", "arguments": "{\"x\":"},
        "_complete": False
    }
}

# Responses API
{
    "fc_123": {
        "id": "call_123",  # from call_id
        "type": "function",
        "function": {"name": "foo", "arguments": "{\"x\":"},
        "_complete": False
    }
}
```

5. **Commonalities to Exploit**:
   - Both accumulate `arguments` string
   - Both need JSON validation for completion
   - Both should use same staging dict structure (just different keys: int vs str)
   - Both should normalize to canonical format before creating ToolUse
   - Completion detection logic can be shared

6. **Key Differences to Handle**:
   - Chat: Track by index (int), deltas have function.arguments
   - Responses: Track by item_id (str), deltas have different event structure
   - Chat: No explicit "done" event, relies on JSON validation
   - Responses: Has explicit `.done` event but still need to validate JSON

**Adjustments Needed to Phase 2 Design**:

1. **Staging Structure Enhancement**:
   - Add `_complete` flag to staging entries
   - Use canonical format in staging (not simplified dict)
   - This allows shared completion detection logic

2. **Completion Detection Enhancement**:
   - Check if arguments became valid JSON
   - Check if tool was previously incomplete (`_complete == False`)
   - Only create ToolUse for "newly complete" tools
   - Mark as complete in staging

3. **Shared Helper Functions**:
   - `_detect_complete_tools()` should work on canonical format
   - Returns newly-completed tools, updates staging with `_complete=True`
   - This function can be shared between both APIs

4. **Normalization Step**:
   - `_update_staging_chat()` creates/updates canonical format
   - `_update_staging_resp()` creates/updates canonical format
   - Both produce same staging structure, just keyed differently

**Action Items**:
- ✅ Verified both APIs are handled
- ✅ Identified commonalities (canonical format, shared completion detection)
- ⚠️ Need to refine staging structure to include `_complete` flag
- ⚠️ Need to ensure completion detection marks tools complete
- ⚠️ Need to ensure we use canonical format throughout

**Refinements Applied**:
- ✅ Updated staging structure to canonical format with `_complete` flag
- ✅ Updated `_detect_complete_tools()` to track newly-complete vs already-complete
- ✅ Updated `_update_staging_chat()` to create/update canonical format
- ✅ Updated `_update_staging_resp()` to create/update canonical format
- ✅ Updated `_create_tool_use()` to extract from canonical format
- ✅ Updated `_extract_tool_calls_chat()` to output canonical format
- ✅ Updated `_extract_tool_calls_resp()` to output canonical format

**Benefits Achieved**:
- ✅ Same completion detection logic works for both APIs
- ✅ Both APIs use identical staging structure (just different keys)
- ✅ Prevents re-processing of already-complete tools
- ✅ Easy conversion to ToolUse (canonical format → ToolUse)
- ✅ Matches OpenAI's internal structure closely

**Next**: Proceed to Phase 3: Detailed Implementation Design
