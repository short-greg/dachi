# Message System: Msg/Resp Patterns

The message system is Dachi's communication layer, built on top of the foundation architecture. It provides a unified interface for working with AI models while handling the complexities of streaming, processing, and state management.

## Core Concepts

### Msg: Complete Message State

`Msg` represents a complete, accumulated message - not incremental updates. This is the format that gets sent to LLMs and represents the canonical state of communication.

```python
from dachi.core import Msg, Attachment

# Basic text message
msg = Msg(role="user", text="Hello, can you help me with Python?")

# Message with attachments (multimodal)
msg = Msg(
    role="user", 
    text="What's in this image?",
    attachments=[
        Attachment(
            kind="image",
            data="base64_encoded_image_data",
            mime="image/png"
        )
    ]
)

# Message with tool calls and results
msg = Msg(
    role="assistant",
    text="I'll search for that information.",
    tool_calls=[
        ToolUse(
            tool_id="search_001",
            option=search_tool,
            inputs=SearchInput(query="Python tutorial"),
            result="Found 10 Python tutorials..."
        )
    ]
)
```

**Key Properties:**
- **Complete state**: Always contains full content, not deltas
- **LLM-ready**: Can be directly sent to any LLM adapter
- **Rich content**: Supports text, attachments, tools, metadata
- **Serializable**: Can be persisted as part of conversation history

### Resp: Response with Accumulation Support

`Resp` represents a complete response from an AI system, with built-in support for streaming accumulation and output processing.

```python
from dachi.core import Resp, RespDelta

# Complete response (non-streaming)
resp = Resp(
    msg=Msg(role="assistant", text="Here's how to use Python..."),
    model="gpt-4",
    usage={"prompt_tokens": 50, "completion_tokens": 100},
    finish_reason="stop"
)

# Access response content
print(resp.msg.text)  # Complete accumulated text
print(resp.model)     # Model that generated response
print(resp.usage)     # Token usage information
```

**Field Architecture:**
- **`msg`**: Complete accumulated message (ready for LLM consumption)
- **`out`**: Processed output value (any type - str, dict, BaseModel, etc.)
- **`delta`**: Current streaming chunk information only
- **`out_store`**: State storage for output processors during streaming
- **`data`**: Internal storage for raw API responses

## Streaming Response Flow

The streaming architecture separates complete state from incremental updates:

### Basic Streaming Pattern

```python
from dachi.proc import OpenAIChat

llm = OpenAIChat()
msg = Msg(role="user", text="Write a short story about AI")

# Streaming accumulates in resp.msg, deltas in resp.delta
for resp in llm.stream(msg):
    # resp.msg.text contains ALL text accumulated so far
    print(f"Full text so far: {resp.msg.text}")
    
    # resp.delta.text contains ONLY the new chunk
    if resp.delta.text:
        print(f"New chunk: '{resp.delta.text}'")
    
    # Check if streaming is complete
    if resp.delta.finish_reason:
        print(f"Streaming finished: {resp.delta.finish_reason}")
        break
```

### Streaming with Output Processing

```python
from dachi.proc import TextOut, WordCountOut

# Multiple output processors
resp_stream = llm.stream(
    msg, 
    out={
        'text': TextOut(),         # Extract clean text
        'word_count': WordCountOut()  # Count words as we go
    }
)

for resp in resp_stream:
    # resp.out contains processed results
    processed_results = resp.out
    print(f"Clean text: {processed_results['text']}")
    print(f"Words so far: {processed_results['word_count']}")
    
    # resp.out_store maintains processor state across chunks
    # (Used internally by processors for accumulation)
```

### Advanced Streaming: Thinking Models

```python
from dachi.proc import OpenAIResp  # For reasoning models

llm = OpenAIResp()  # Supports thinking/reasoning
msg = Msg(role="user", text="Solve this complex math problem...")

for resp in llm.stream(msg):
    # Regular response text
    if resp.delta.text:
        print(f"Response: {resp.delta.text}", end="", flush=True)
    
    # Model's internal reasoning (o1-style models)
    if resp.delta.thinking:
        print(f"\n[Thinking: {resp.delta.thinking}]", end="", flush=True)
    
    # Complete accumulated content
    print(f"\nFull response: {resp.msg.text}")
    print(f"Full thinking: {resp.thinking}")
```

## Response Processing Patterns

### Single Output Processor

```python
from dachi.proc import JsonOut

# Process response into structured data
resp = llm.forward(
    msg,
    out=JsonOut(schema=UserProfile)  # Parse JSON into Pydantic model
)

# resp.out contains the parsed UserProfile object
user_profile = resp.out
print(f"Name: {user_profile.name}")
print(f"Age: {user_profile.age}")
```

### Multiple Output Processors

```python
# Process response in multiple ways simultaneously
resp = llm.forward(
    msg,
    out={
        'summary': SummaryOut(max_length=50),
        'sentiment': SentimentOut(),
        'keywords': KeywordExtractorOut(count=5),
        'structured': JsonOut(schema=ResponseData)
    }
)

# Each processor produces its own output
results = resp.out
print(f"Summary: {results['summary']}")
print(f"Sentiment: {results['sentiment']}")
print(f"Keywords: {results['keywords']}")
print(f"Structured: {results['structured']}")
```

### Tuple Output Processing

```python
# Multiple processors in tuple form
resp = llm.forward(
    msg,
    out=(TextOut(), SentimentOut(), KeywordExtractorOut())
)

# resp.out is a tuple with results in order
text, sentiment, keywords = resp.out
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")  
print(f"Keywords: {keywords}")
```

## Streaming State Management

### The spawn() Pattern

During streaming, `spawn()` creates new responses while preserving processing state:

```python
# Internal streaming implementation (simplified)
class StreamingAdapter:
    def stream(self, msg):
        prev_resp = None
        accumulated_text = ""
        
        for chunk_data in api_stream:
            # Accumulate text
            chunk_text = chunk_data.get("content", "")
            accumulated_text += chunk_text
            
            # Create complete message state
            complete_msg = Msg(role="assistant", text=accumulated_text)
            
            if prev_resp is None:
                # First chunk
                resp = Resp(msg=complete_msg)
                resp.delta = RespDelta(text=chunk_text)
            else:
                # Subsequent chunks - preserve processing state
                resp = prev_resp.spawn(
                    msg=complete_msg,  # Updated complete state
                    data=chunk_data    # Raw chunk data
                )
                resp.delta = RespDelta(text=chunk_text)  # New delta
            
            # resp.out_store preserved across chunks for processors
            # resp.out updated with current processed results
            
            prev_resp = resp
            yield resp
```

### Processor State Continuity

```python
# Custom output processor that maintains state across chunks
class RunningAverageOut(ToOut):
    def delta(self, resp: Resp, state: dict, is_last: bool):
        # Use resp.out_store[key] for persistent state across chunks
        if 'count' not in state:
            state['count'] = 0
            state['sum'] = 0
        
        # Process current chunk
        numbers = extract_numbers(resp.delta.text or "")
        for num in numbers:
            state['sum'] += num
            state['count'] += 1
        
        # Return current average
        return state['sum'] / state['count'] if state['count'] > 0 else 0

# Usage
resp_stream = llm.stream(msg, out=RunningAverageOut())
for resp in resp_stream:
    print(f"Running average: {resp.out}")
    # Processor state automatically maintained in resp.out_store
```

## Dialog: Conversation Management

`Dialog` is Dachi's system for managing multi-turn conversations. It provides both simple linear conversations and complex tree-structured dialogs for advanced use cases.

### BaseDialog: The Foundation

All dialog types inherit from `BaseDialog`, providing consistent interface patterns:

```python
from dachi.core import BaseDialog

# Common operations available on all dialog types
dialog = SomeDialog()

# Iterate over messages
for msg in dialog:
    print(f"{msg.role}: {msg.text}")

# Access by index
first_msg = dialog[0]
dialog[1] = Msg(role="system", text="Updated message")

# Manage conversations
dialog.append(Msg(role="user", text="Hello"))
dialog.extend([msg1, msg2, msg3])
dialog.remove(unwanted_msg)

# Convert and render
message_list = dialog.aslist()
conversation_text = dialog.render()
```

### ListDialog: Linear Conversations

The most common dialog type for straightforward conversations:

```python
from dachi.core import ListDialog

# Build conversation step by step
dialog = ListDialog()
dialog.append(Msg(role="system", text="You are a helpful assistant"))
dialog.append(Msg(role="user", text="Hello"))

# Send to LLM and continue conversation
resp = llm.forward(dialog)
dialog.append(resp.msg)

# Add user follow-up
dialog.append(Msg(role="user", text="Can you help me with Python?"))
resp = llm.forward(dialog)
dialog.append(resp.msg)

print(f"Conversation has {len(dialog)} messages")
```

### TreeDialog: Branching Conversations

For complex scenarios where conversations can branch and explore multiple paths:

```python
from dachi.core import TreeDialog

# Create tree-structured dialog
dialog = TreeDialog()
dialog.append(Msg(role="user", text="I need help with a project"))

# Get initial response
resp1 = llm.forward(dialog)
dialog.append(resp1.msg)

# Now explore multiple conversation branches
# Branch 1: Technical details
dialog.append(Msg(role="user", text="Tell me about the technical requirements"))
resp_tech = llm.forward(dialog)
dialog.append(resp_tech.msg)

# Go back and explore Branch 2: Timeline
dialog.rise(2)  # Go back 2 steps to branch point
dialog.append(Msg(role="user", text="What's a realistic timeline?"))
resp_timeline = llm.forward(dialog)
dialog.append(resp_timeline.msg)

# Navigate the tree structure
print(f"Current path indices: {dialog.indices}")
print(f"Available branches at each level: {dialog.counts}")
```

### Dialog Manipulation

Powerful tools for working with conversation history:

```python
from dachi.core import include_role, exclude_role, exclude_messages

# Filter by role
user_messages = include_role(dialog, "user")
ai_responses = include_role(dialog, "assistant") 
system_messages = include_role(dialog, "system")

# Multiple roles
human_conversation = include_role(dialog, "user", "assistant")

# Exclude unwanted content
no_system = exclude_role(dialog, "system")
no_tools = exclude_role(dialog, "tool")

# Custom filtering
important_only = exclude_messages(dialog, lambda msg: msg.filtered)
recent_only = ListDialog([msg for msg in dialog if msg.timestamp > recent_cutoff])
```

### Dialog Patterns for AI Systems

#### Context Management
```python
class ContextualAgent:
    def __init__(self):
        self.dialog = ListDialog()
        self.context_window = 10  # Keep last 10 messages
        
    def chat(self, user_input: str) -> str:
        # Add user message
        self.dialog.append(Msg(role="user", text=user_input))
        
        # Manage context window
        if len(self.dialog) > self.context_window:
            # Keep system message + recent messages
            system_msgs = include_role(self.dialog, "system")
            recent_msgs = ListDialog(list(self.dialog)[-self.context_window:])
            self.dialog = ListDialog()
            self.dialog.extend(system_msgs[:1])  # Keep first system message
            self.dialog.extend(recent_msgs)
        
        # Generate response
        resp = self.llm.forward(self.dialog)
        self.dialog.append(resp.msg)
        
        return resp.msg.text
```

#### Multi-Agent Conversations
```python
class MultiAgentDialog:
    def __init__(self):
        self.dialog = ListDialog()
        self.agents = {
            "researcher": ResearchAgent(),
            "writer": WritingAgent(), 
            "critic": CriticAgent()
        }
    
    def collaborative_task(self, task: str):
        # Initialize task
        self.dialog.append(Msg(role="user", text=task))
        
        # Agent collaboration loop
        for round in range(3):
            for agent_name, agent in self.agents.items():
                # Each agent sees full conversation
                resp = agent.process(self.dialog)
                
                # Add agent response with role identification
                self.dialog.append(Msg(
                    role="assistant",
                    text=resp.msg.text,
                    alias=agent_name  # Track which agent responded
                ))
        
        return self.dialog.render()
```

#### Conversation Branching Strategy
```python
class ExploratoryAgent:
    def __init__(self):
        self.tree = TreeDialog()
        
    def explore_topic(self, initial_query: str):
        self.tree.append(Msg(role="user", text=initial_query))
        
        # Get initial response
        resp = self.llm.forward(self.tree)
        self.tree.append(resp.msg)
        
        # Explore multiple angles
        exploration_paths = [
            "Can you elaborate on the technical aspects?",
            "What are the practical applications?",
            "What are the potential risks or limitations?"
        ]
        
        results = {}
        for i, path_query in enumerate(exploration_paths):
            if i > 0:
                # Return to branching point
                self.tree.rise(1)
            
            # Explore this path
            self.tree.append(Msg(role="user", text=path_query))
            resp = self.llm.forward(self.tree)
            self.tree.append(resp.msg)
            
            results[f"path_{i}"] = resp.msg.text
        
        return results
```

### Dialog Rendering and Export

```python
# Custom rendering
class FormattedRenderer:
    def __call__(self, dialog: BaseDialog) -> str:
        result = []
        for i, msg in enumerate(dialog):
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            name = msg.alias or msg.role.title()
            result.append(f"[{timestamp}] {name}: {msg.text}")
        return "\n".join(result)

# Use custom renderer
dialog._renderer = FormattedRenderer()
formatted_output = dialog.render()

# Export for analysis
conversation_data = {
    "messages": [msg.model_dump() for msg in dialog],
    "metadata": {
        "total_messages": len(dialog),
        "participants": list(set(msg.role for msg in dialog)),
        "duration": dialog[-1].timestamp - dialog[0].timestamp
    }
}
```

### Dialog Best Practices

#### 1. Context Window Management
```python
# ✅ Good - actively manage context size
def manage_context(dialog: ListDialog, max_tokens: int = 4000):
    estimated_tokens = sum(len(msg.text.split()) for msg in dialog)
    
    if estimated_tokens > max_tokens:
        # Keep system messages + recent conversation
        system_msgs = include_role(dialog, "system")
        recent_msgs = dialog[-10:]  # Last 10 messages
        
        new_dialog = ListDialog()
        new_dialog.extend(system_msgs)
        new_dialog.extend(recent_msgs)
        return new_dialog
    
    return dialog

# ❌ Bad - letting context grow indefinitely  
# This will eventually hit token limits
```

#### 2. Role Consistency
```python
# ✅ Good - consistent role usage
dialog.append(Msg(role="user", text=user_input))
dialog.append(Msg(role="assistant", text=ai_response))
dialog.append(Msg(role="system", text=system_instruction))

# ❌ Bad - inconsistent or custom roles without handling
dialog.append(Msg(role="customer", text=input))  # Non-standard role
dialog.append(Msg(role="ai_agent", text=response))  # Might not work with LLMs
```

#### 3. Conversation State Tracking
```python
# ✅ Good - track conversation state in dialog metadata
dialog.append(Msg(
    role="assistant", 
    text=response,
    meta={
        "confidence": 0.95,
        "sources": ["doc1.pdf", "doc2.pdf"],
        "reasoning_steps": 3
    }
))

# ❌ Bad - losing important conversation context
# Just storing raw text without metadata
```

The Dialog system provides flexible conversation management that scales from simple chat interfaces to complex multi-agent systems with branching conversation trees.

## Message System Best Practices

### 1. Always Use Complete State
```python
# ✅ Good - msg.text contains complete accumulated text
for resp in llm.stream(msg):
    complete_response = resp.msg.text  # Full response so far
    
# ❌ Bad - don't accumulate deltas manually
accumulated = ""
for resp in llm.stream(msg):
    accumulated += resp.delta.text or ""  # Manual accumulation
```

### 2. Separate Processing from Content
```python
# ✅ Good - use output processors for processing
resp = llm.forward(msg, out=JsonOut(schema=MySchema))
structured_data = resp.out  # Processed result
original_text = resp.msg.text  # Original response

# ❌ Bad - don't mix content and processing
raw_response = llm.forward(msg)
try:
    structured_data = json.loads(resp.msg.text)  # Manual parsing
except:
    # Error handling...
```

### 3. Leverage State Continuity
```python
# ✅ Good - let the system manage processor state
custom_processor = ComplexAnalyzer()
for resp in llm.stream(msg, out=custom_processor):
    analysis = resp.out  # Automatically updated state
    
# ❌ Bad - don't manage processor state manually
analyzer_state = {}
for resp in llm.stream(msg):
    analyzer_state = update_analysis(analyzer_state, resp.delta.text)
```

The message system provides a clean, consistent interface for AI communication while handling the complexities of streaming, state management, and output processing behind the scenes. This enables building sophisticated intelligent systems without getting bogged down in low-level API details.