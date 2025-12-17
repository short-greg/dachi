# Communication & Request Handling in Dachi

This guide covers Dachi's communication and request handling infrastructure, essential for building multi-agent systems, behavior trees, and concurrent AI workflows.

## Overview

Dachi provides three core communication primitives in the `dachi.act.comm` module:

- **Bulletin**: Message board for inter-agent communication and task coordination
- **Blackboard**: Shared state storage with reactive callbacks and TTL support
- **AsyncDispatcher**: Centralized async request dispatcher with concurrency control
- **Buffer/BufferIter**: Thread-safe streaming data collection and processing

These components work together to enable sophisticated AI workflows with proper thread safety, resource management, and coordination patterns.

## Component Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Bulletin  │    │  Blackboard  │    │   Buffer    │
│ (Messages)  │    │   (State)    │    │ (Streaming) │
└──────┬──────┘    └──────┬───────┘    └──────┬──────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                ┌─────────┴──────────┐
                │  AsyncDispatcher   │
                │ (Request Handling) │
                └────────────────────┘
```

## Core Components

### Bulletin: Inter-Agent Message Passing

The Bulletin acts as a message board where agents can post messages, retrieve work items, and coordinate tasks. It supports scoping for multi-tenant safety and callbacks for event-driven patterns.

#### Basic Usage

```python
from dachi.act.comm import Bulletin
from pydantic import BaseModel

class TaskRequest(BaseModel):
    task_id: str
    priority: int
    description: str
    assigned_to: str = None

# Create bulletin for task coordination
task_board = Bulletin[TaskRequest]()

# Agent A posts a task
task = TaskRequest(
    task_id="task_001", 
    priority=8, 
    description="Process customer data"
)
post_id = task_board.publish(task, lock=True)  # Lock prevents concurrent access

# Agent B retrieves and processes the task
post = task_board.retrieve_first(
    filter_func=lambda t: t.priority >= 5,  # High priority only
    include_locked=False  # Available tasks only
)

if post:
    task_item = post["item"]
    print(f"Processing: {task_item.description}")
    
    # Process the task...
    
    # Mark complete by removing
    task_board.remove(post["id"])
```

#### Scoped Communication

```python
# Each agent operates in its own scope
agent_id = "agent_alpha"

# Publish within scope (prevents cross-contamination)
task_board.publish(task, scope=agent_id)

# Retrieve only from own scope
my_tasks = task_board.retrieve_all(scope=agent_id)

# Other agents won't see scoped messages
agent_beta_tasks = task_board.retrieve_all(scope="agent_beta")  # Empty
```

#### Event-Driven Patterns

```python
def task_monitor(post, event_type):
    if event_type == Bulletin.ON_PUBLISH:
        print(f"New task posted: {post['item'].task_id}")
    elif event_type == Bulletin.ON_EXPIRE:
        print(f"Task expired: {post['item'].task_id}")

task_board.register_callback(task_monitor)

# Tasks with automatic expiration
task_board.publish(urgent_task, ttl=300)  # 5 minutes
```

### Blackboard: Shared State Management

The Blackboard provides thread-safe key-value storage for shared state between agents, behavior tree nodes, and system components. Supports TTL, scoping, and reactive callbacks.

#### Basic State Operations

```python
from dachi.act.comm import Blackboard

# Global blackboard instance
blackboard = Blackboard()

# Simple attribute access
blackboard.mission_status = "in_progress"
blackboard.agent_count = 5

# Safe access with defaults
status = blackboard.get("mission_status", "unknown")
count = blackboard.get("agent_count", 0)

# Check existence
if blackboard.has("mission_status"):
    print("Mission is active")
```

#### Scoped State Management

```python
robot_id = "robot_001"

# Store robot-specific state
blackboard.set_with_ttl("battery_level", 85, 60, scope=robot_id)
blackboard.set_with_ttl("location", {"x": 100, "y": 200}, 30, scope=robot_id)

# Retrieve robot state
battery = blackboard.get("battery_level", scope=robot_id)
location = blackboard.get("location", scope=robot_id)

# List all keys in scope
robot_keys = blackboard.keys(scope=robot_id)
```

#### Reactive State Changes

```python
def state_monitor(key, value, event_type):
    if event_type == Blackboard.ON_SET:
        print(f"State updated: {key} = {value}")
    elif event_type == Blackboard.ON_EXPIRE:
        print(f"State expired: {key}")

blackboard.register_callback(state_monitor)

# These trigger callbacks
blackboard.alert_level = "high"  # Triggers ON_SET
blackboard.set_with_ttl("temp_cache", "data", 5)  # Auto-expires
```

### AsyncDispatcher: Concurrent Request Management

The AsyncDispatcher manages concurrent AI requests with concurrency limiting, state tracking, and streaming support. Essential for behavior trees that need non-blocking AI operations.

#### Non-Streaming Requests

```python
from dachi.act.comm import AsyncDispatcher, RequestState
from dachi.proc import LangModel  # Example: your LLM adapter

dispatcher = AsyncDispatcher(max_concurrency=5)
llm = my_llm_adapter  # Your LangModel instance

# Submit non-blocking request
req_id = dispatcher.submit_proc(
    llm,
    {"role": "user", "content": "Summarize this document"},
    temperature=0.7
)

# Poll for completion (behavior tree pattern)
status = dispatcher.status(req_id)
if status.state == RequestState.DONE:
    result = dispatcher.result(req_id)
    print(f"AI Response: {result}")
elif status.state == RequestState.ERROR:
    print("Request failed")
else:
    print("Still processing...")
```

#### Streaming Requests

```python
# Submit streaming request
req_id = dispatcher.submit_stream(
    streaming_llm,
    {"role": "user", "content": "Write a story"},
    temperature=0.8
)

# Consume stream
accumulated_text = ""
for chunk in dispatcher.stream_result(req_id):
    if hasattr(chunk, 'delta') and chunk.delta.text:
        accumulated_text += chunk.delta.text
        print(chunk.delta.text, end="", flush=True)

print(f"\nFinal story: {accumulated_text}")
```

#### Callback-Based Completion

```python
def on_completion(req_id, result, error):
    if error:
        print(f"Request {req_id} failed: {error}")
    else:
        print(f"Request {req_id} completed: {result}")

# Submit with callback
req_id = dispatcher.submit_proc(
    llm, 
    message,
    _callback=on_completion
)
```

### Buffer: Streaming Data Collection

Buffer provides thread-safe collection and processing of streaming data, commonly used for accumulating AI response tokens or processing data pipelines.

#### Basic Streaming

```python
from dachi.act.comm import Buffer

# Create buffer for streaming responses
response_buffer = Buffer(buffer=[])

# Simulate streaming data collection
for chunk in llm_stream_response():
    if response_buffer.is_open():
        response_buffer.add(chunk.text)

response_buffer.close()

# Get complete response
full_response = ''.join(response_buffer.get())
```

#### Iterator-Based Processing

```python
# Process stream with iterator
iterator = response_buffer.it()

# Sequential processing
processed_chunks = []
while not iterator.end():
    try:
        chunk = iterator.read()
        processed_chunks.append(transform_chunk(chunk))
    except StopIteration:
        break

# Batch processing remaining items
remaining = iterator.read_all()
batch_processed = process_batch(remaining)

# Functional operations
total_length = iterator.read_reduce(lambda acc, chunk: acc + len(chunk), 0)
uppercase_chunks = list(iterator.read_map(str.upper))
```

## Integration Patterns

### Behavior Tree Integration

```python
from dachi.act import Task, TaskStatus

class LLMQueryTask(Task):
    """Behavior tree task that performs non-blocking LLM queries."""
    
    def __init__(self, dispatcher, blackboard, llm, query):
        self.dispatcher = dispatcher
        self.blackboard = blackboard
        self.llm = llm
        self.query = query
        self.req_id = None
        
    def tick(self) -> TaskStatus:
        if self.req_id is None:
            # First tick: dispatch request
            self.req_id = self.dispatcher.submit_proc(self.llm, self.query)
            self.blackboard.set_with_ttl("last_request", self.req_id, 300)
            return TaskStatus.RUNNING
            
        # Subsequent ticks: check status
        status = self.dispatcher.status(self.req_id)
        if status.state == RequestState.DONE:
            result = self.dispatcher.result(self.req_id)
            self.blackboard.llm_response = result
            return TaskStatus.SUCCESS
        elif status.state == RequestState.ERROR:
            self.blackboard.llm_error = "Query failed"
            return TaskStatus.FAILURE
        else:
            return TaskStatus.RUNNING  # Still processing

class CheckResourcesCondition(Task):
    """Condition that checks shared state."""
    
    def __init__(self, blackboard, robot_id):
        self.blackboard = blackboard
        self.robot_id = robot_id
        
    def tick(self) -> TaskStatus:
        battery = self.blackboard.get("battery_level", 0, scope=self.robot_id)
        fuel = self.blackboard.get("fuel_level", 0, scope=self.robot_id)
        
        if battery > 30 and fuel > 20:
            self.blackboard.set_with_ttl("ready_for_mission", True, 300)
            return TaskStatus.SUCCESS
        else:
            return TaskStatus.FAILURE
```

### Multi-Agent Workflow

```python
class AgentCoordinator:
    """Coordinates multiple agents using communication primitives."""
    
    def __init__(self):
        self.task_board = Bulletin[TaskRequest]()
        self.blackboard = Blackboard()
        self.dispatcher = AsyncDispatcher()
        self.agents = {}
        
    def setup_monitoring(self):
        """Setup event monitoring."""
        def task_monitor(post, event):
            if event == Bulletin.ON_PUBLISH:
                self.blackboard.pending_tasks = self.blackboard.get("pending_tasks", 0) + 1
            elif event == Bulletin.ON_REMOVE:
                self.blackboard.completed_tasks = self.blackboard.get("completed_tasks", 0) + 1
                
        def state_monitor(key, value, event):
            if key == "alert_level" and event == Blackboard.ON_SET:
                self.broadcast_alert(value)
                
        self.task_board.register_callback(task_monitor)
        self.blackboard.register_callback(state_monitor)
    
    def submit_task(self, task: TaskRequest) -> str:
        """Submit task for agent processing."""
        return self.task_board.publish(task, lock=True)
    
    def get_available_work(self, agent_id: str) -> TaskRequest:
        """Get work for a specific agent."""
        post = self.task_board.retrieve_first(
            filter_func=lambda t: t.assigned_to is None or t.assigned_to == agent_id,
            order_func=lambda t: -t.priority,  # Highest priority first
            scope=agent_id
        )
        
        if post:
            # Assign to agent and release lock
            post["item"].assigned_to = agent_id
            self.task_board.release(post["id"])
            return post["item"]
        return None
    
    def update_agent_status(self, agent_id: str, status: dict):
        """Update agent status with TTL."""
        self.blackboard.set_with_ttl(
            "status", 
            status, 
            60,  # 1 minute TTL
            scope=agent_id
        )
    
    def broadcast_alert(self, alert_level: str):
        """Broadcast alert to all agents."""
        alert = TaskRequest(
            task_id=f"alert_{int(time.time())}",
            priority=10,
            description=f"System alert: {alert_level}"
        )
        
        for agent_id in self.agents:
            self.task_board.publish(alert, scope=agent_id)
```

### Streaming Pipeline

```python
class StreamingPipeline:
    """Pipeline for processing streaming AI responses."""
    
    def __init__(self):
        self.dispatcher = AsyncDispatcher()
        self.blackboard = Blackboard()
        
    def process_streaming_request(self, llm, query):
        """Process streaming request with real-time updates."""
        # Submit streaming request
        req_id = self.dispatcher.submit_stream(llm, query)
        
        # Create buffer for accumulation
        buffer = Buffer(buffer=[])
        
        # Process stream
        chunk_count = 0
        for chunk in self.dispatcher.stream_result(req_id):
            buffer.add(chunk.delta.text or "")
            chunk_count += 1
            
            # Update progress in blackboard
            self.blackboard.set_with_ttl(
                "streaming_progress",
                {"chunks_received": chunk_count, "req_id": req_id},
                30
            )
            
            # Real-time processing
            if chunk_count % 10 == 0:  # Every 10 chunks
                partial_text = ''.join(buffer.get())
                self.process_partial_result(partial_text)
        
        # Final processing
        buffer.close()
        final_text = ''.join(buffer.get())
        
        # Store final result
        self.blackboard.final_result = final_text
        
        return final_text
    
    def process_partial_result(self, partial_text: str):
        """Process partial results for real-time analysis."""
        # Example: sentiment analysis on partial text
        if len(partial_text) > 100:
            sentiment = analyze_sentiment(partial_text)
            self.blackboard.set_with_ttl("current_sentiment", sentiment, 10)
```

## Thread Safety & Best Practices

### Thread Safety Guarantees

All communication components are thread-safe:

- **Bulletin**: Uses internal RLock for all operations
- **Blackboard**: Uses internal RLock for state management  
- **AsyncDispatcher**: Thread-safe job management with proper synchronization
- **Buffer**: Thread-safe data collection (single producer recommended)

### Scoping Best Practices

1. **Use unique scopes** for multi-agent systems:
   ```python
   agent_scope = f"agent_{agent_id}"
   blackboard.set_with_ttl("status", data, 60, scope=agent_scope)
   ```

2. **Namespace bulletins** by function:
   ```python
   task_board = Bulletin[TaskRequest]()  # For work distribution
   alert_board = Bulletin[Alert]()       # For alerts
   ```

3. **Consistent scope naming**:
   ```python
   # Good: consistent patterns
   f"agent_{id}", f"task_{id}", f"session_{id}"
   
   # Avoid: inconsistent naming
   f"{id}_agent", f"task-{id}", f"session.{id}"
   ```

### Resource Management

1. **Clean up expired data**:
   ```python
   # Blackboard automatically cleans expired keys
   blackboard.set_with_ttl("temp_data", value, 300)
   
   # Bulletin automatic cleanup on retrieval
   bulletin.retrieve_all()  # Triggers cleanup
   ```

2. **Monitor dispatcher concurrency**:
   ```python
   dispatcher = AsyncDispatcher(max_concurrency=5)  # Prevent API overload
   ```

3. **Handle streaming properly**:
   ```python
   # Always consume streams fully
   for chunk in dispatcher.stream_result(req_id):
       process(chunk)
   # Stream auto-cleaned after consumption
   ```

### Error Handling

```python
# Robust bulletin operations
post = bulletin.retrieve_first(id=post_id)
if post is None:
    logger.warning(f"Post {post_id} not found or expired")
    return

# Safe blackboard access
value = blackboard.get("key", default_value)

# Handle dispatcher errors
status = dispatcher.status(req_id)
if status and status.state == RequestState.ERROR:
    logger.error(f"Request failed: {status.error}")
```

## When to Use Each Component

| Component | Use When | Example Use Cases |
|-----------|----------|-------------------|
| **Bulletin** | Inter-agent messaging, task queues, event notifications | Work distribution, agent coordination, alert broadcasting |
| **Blackboard** | Shared state, configuration, caching | Robot sensor data, mission parameters, system status |  
| **AsyncDispatcher** | Non-blocking AI operations, concurrent processing | LLM queries in behavior trees, parallel API calls |
| **Buffer** | Streaming data, token accumulation, pipeline processing | LLM streaming responses, real-time data processing |

## Common Pitfalls & Solutions

### Pitfall: Race Conditions in Bulletin

**Problem**: Multiple agents grabbing the same locked item
```python
# BAD: Race condition
post = bulletin.retrieve_first()
if not post["locked"]:  # Another agent might lock it here
    bulletin.remove(post["id"])  # Error!
```

**Solution**: Use proper locking patterns
```python
# GOOD: Atomic operations
post = bulletin.retrieve_first(include_locked=False)
if post:
    bulletin.remove(post["id"])  # Safe
```

### Pitfall: Blackboard Scope Leakage

**Problem**: Data visible across agents
```python
# BAD: No scoping
blackboard.agent_status = "active"  # Visible to all agents

# BAD: Inconsistent scoping
blackboard.set_with_ttl("status", data, 60, scope="agent1")
other_status = blackboard.agent_status  # Wrong scope
```

**Solution**: Consistent scoping
```python
# GOOD: Always use scopes for agent data
agent_id = "agent_001"
blackboard.set_with_ttl("status", "active", 60, scope=agent_id)
status = blackboard.get("status", scope=agent_id)
```

### Pitfall: Stream Not Consumed

**Problem**: Streaming requests not cleaned up
```python
# BAD: Stream created but not consumed
req_id = dispatcher.submit_stream(llm, query)
# Stream hangs, resources leaked
```

**Solution**: Always consume streams
```python
# GOOD: Full consumption
req_id = dispatcher.submit_stream(llm, query)
chunks = list(dispatcher.stream_result(req_id))
# Stream automatically cleaned up
```

### Pitfall: Blocking Behavior Trees

**Problem**: Synchronous AI calls in behavior tree tasks
```python
# BAD: Blocks entire behavior tree
class BadLLMTask(Task):
    def tick(self):
        result = llm.forward(query)  # Blocks!
        return TaskStatus.SUCCESS
```

**Solution**: Use AsyncDispatcher pattern
```python
# GOOD: Non-blocking pattern
class GoodLLMTask(Task):
    def tick(self):
        if self.req_id is None:
            self.req_id = dispatcher.submit_proc(llm, query)
            return TaskStatus.RUNNING
            
        status = dispatcher.status(self.req_id)
        if status.state == RequestState.DONE:
            self.result = dispatcher.result(self.req_id)
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING
```

## Performance Considerations

- **Concurrency**: Tune AsyncDispatcher `max_concurrency` based on API limits
- **TTL**: Use appropriate TTL values to prevent memory leaks
- **Scoping**: Scoped operations are slightly slower but prevent data conflicts
- **Callbacks**: Keep callback functions lightweight to avoid blocking
- **Buffer Size**: Monitor buffer growth in long-running streams

## Conclusion

Dachi's communication and request handling infrastructure provides a robust foundation for building sophisticated AI systems. The combination of Bulletin (messaging), Blackboard (state), AsyncDispatcher (requests), and Buffer (streaming) enables clean separation of concerns while maintaining thread safety and performance.

Key takeaways:
- Use scoping for multi-agent safety
- Leverage async patterns for non-blocking AI operations  
- Monitor resource usage and implement proper cleanup
- Follow thread-safe patterns for concurrent access
- Choose the right component for each use case

For more examples and advanced usage patterns, see the test files in `tests/comm/` and the API documentation.