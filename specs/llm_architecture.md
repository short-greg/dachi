# Dachi LLM Architecture Documentation

## Overview

Dachi provides a unified framework for working with different LLM providers (OpenAI, Claude, Gemini, etc.) through a standardized pipeline. The architecture separates concerns into distinct layers:

1. **Unified Message/Response Format** (`Msg`/`Resp`)
2. **Provider Adapters** (`AIAdapt`)
3. **Response Processors** (`RespProc`)
4. **Output Converters** (`ToOut`)

## Core Components

### 1. Msg (Unified Message Format)

The `Msg` class represents a standardized message format that works across all LLM providers:

```python
from dachi.core import Msg, Attachment

# Basic text message
msg = Msg(
    role="user",
    text="Hello, how are you?"
)

# Message with attachments (images, etc.)
msg = Msg(
    role="user", 
    text="What do you see in this image?",
    attachments=[
        Attachment(
            kind="image",
            data="data:image/png;base64,iVBOR...", 
            mime="image/png"
        )
    ]
)

# Tool output message
msg = Msg(
    role="tool",
    text="The weather is 72°F and sunny",
    tool_outs=[
        ToolOut(
            tool_call_id="call_123",
            option=weather_tool,
            result="72°F, sunny"
        )
    ]
)
```

### 2. Resp (Unified Response Format)

The `Resp` class provides a standardized response format with core fields and streaming support:

```python
from dachi.core import Resp, RespDelta

# Complete response
resp = Resp(
    text="Hello! I'm doing well, thank you.",
    thinking="The user is greeting me politely...",
    finish_reason="stop",
    usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    model="gpt-4o",
    response_id="resp_123"
)

# Streaming response with delta
resp = Resp(
    delta=RespDelta(
        text="Hello! I'm",
        finish_reason=None
    )
)

# Access unified fields across providers
tokens_used = resp.usage.get("total_tokens", 0)
reasoning = resp.thinking
content = resp.text
```

### 3. AIAdapt (Provider Adapters)

Adapters convert between the unified format and provider-specific APIs:

```python
from dachi.core import OpenAIChat, OpenAIResp

# OpenAI Chat Completions adapter
chat_adapter = OpenAIChat()

# Convert unified format to OpenAI API format
openai_input = chat_adapter.to_input(msg, temperature=0.7, max_tokens=100)
# Returns: {"messages": [...], "temperature": 0.7, "max_tokens": 100}

# Make API call (handled by llm_forward/llm_stream)
openai_response = openai_client.chat.completions.create(**openai_input)

# Convert back to unified format
unified_resp = chat_adapter.from_output(openai_response)
# Returns: Resp with text, usage, model, etc. populated

# Streaming conversion
streaming_resp = chat_adapter.from_streamed(chunk, prev_resp)
```

### 4. RespProc (Response Processors)

Process `Resp` objects to extract and transform data:

```python
from dachi.proc import TextConv, StructConv, ParsedConv

# Extract text content
text_proc = TextConv(name='content', from_='text')
processed_resp = text_proc(resp, is_streamed=False, is_last=True)
# Accumulates text and updates resp.msg.text

# Extract structured JSON
struct_proc = StructConv(name='json_data', from_='text')
processed_resp = struct_proc(resp, is_streamed=False, is_last=True)
# Parses JSON from text content

# Validate against Pydantic model
class UserInfo(BaseModel):
    name: str
    age: int

parsed_proc = ParsedConv(struct=UserInfo, name='user', from_='text')
processed_resp = parsed_proc(resp, is_streamed=False, is_last=True)
# Returns validated UserInfo instance
```

### 5. ToOut (Output Converters)

Convert processed data to specific output formats:

```python
from dachi.proc import ToolExecConv, StrOut, PydanticOut

# Execute tool calls
tool_exec = ToolExecConv(name='tool_results', from_='tool')
results = tool_exec(resp, is_streamed=False, is_last=True)
# Executes tool calls and returns results

# Convert to string
str_out = StrOut(name='output', from_='content')
string_result = str_out(resp, is_streamed=False, is_last=True)

# Convert to Pydantic model
pydantic_out = PydanticOut(out_cls=UserInfo, name='user', from_='content')
model_instance = pydantic_out(resp, is_streamed=False, is_last=True)
```

## Complete Pipeline Example

Here's how all components work together:

```python
from dachi.core import Msg, OpenAIChat
from dachi.proc import TextConv, StructConv
from dachi.proc import llm_forward
import openai

# 1. Create unified message
msg = Msg(role="user", text="List 3 Python web frameworks as JSON")

# 2. Set up adapter and processors  
adapter = OpenAIChat()
text_proc = TextConv()
struct_proc = StructConv()

# 3. Process through pipeline
resp = llm_forward(
    openai.chat.completions.create,
    msg,
    _adapt=adapter,
    _proc=[text_proc, struct_proc],
    model="gpt-4o",
    temperature=0.7
)

# 4. Access results
print(f"Raw text: {resp.text}")
print(f"Structured data: {resp.data['content']}")
print(f"Tokens used: {resp.usage['total_tokens']}")
print(f"Model: {resp.model}")
```

## Streaming Pipeline Example

```python
# Streaming with the same components
for resp_chunk in llm_stream(
    openai.chat.completions.create,
    msg,
    _adapt=adapter,
    _proc=[text_proc],
    model="gpt-4o",
    stream=True
):
    print(f"Delta: {resp_chunk.delta.text}")
    print(f"Accumulated: {resp_chunk.text}")
```

## Key Benefits

### 1. Provider Independence
```python
# Same code works with any provider
claude_adapter = ClaudeAdapt()  # When implemented
gemini_adapter = GeminiAdapt()  # When implemented

# All use the same interface
for adapter in [OpenAIChat(), claude_adapter, gemini_adapter]:
    resp = llm_forward(provider_client.create, msg, _adapt=adapter, _proc=[text_proc])
    print(resp.text)  # Same access pattern
```

### 2. Composable Processing
```python
# Mix and match processors
processors = [
    TextConv(),           # Extract text
    StructConv(),         # Parse JSON
    ToolExecConv(),       # Execute tools
]

resp = llm_forward(api_call, msg, _adapt=adapter, _proc=processors)
```

### 3. Unified Access Patterns
```python
# Same fields across all providers
def analyze_response(resp: Resp):
    print(f"Text: {resp.text}")
    print(f"Reasoning: {resp.thinking}")  
    print(f"Tokens: {resp.usage['total_tokens']}")
    print(f"Model: {resp.model}")
    print(f"Citations: {resp.citations}")
    
# Works regardless of underlying provider
```

## Data Flow Diagram

```
User Input (Msg)
    ↓
AIAdapt.to_input() → Provider API Format
    ↓
API Call → Raw Provider Response  
    ↓
AIAdapt.from_output() → Unified Resp
    ↓
RespProc.delta() → Processed Data
    ↓
ToOut.delta() → Final Output Format
    ↓
Application Use
```

## Extension Points

### Adding New Providers
1. Implement `AIAdapt` subclass
2. Override `to_input()`, `from_output()`, `from_streamed()`
3. Map provider response to unified `Resp` format

### Adding New Processors  
1. Extend `RespProc` for response processing
2. Extend `ToOut` for output conversion
3. Use `from_` field to specify data source

### Custom Response Fields
- Core fields: `text`, `tool`, `thinking`, `citations`, etc.
- Provider-specific data goes in `resp.meta`
- Raw response always available in `resp._data`

This architecture provides a clean separation of concerns while maintaining flexibility for different LLM providers and use cases.