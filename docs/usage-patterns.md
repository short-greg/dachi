# Usage Patterns

This guide demonstrates the canonical usage patterns for Dachi's core communication abstractions. These patterns show how to effectively use Blackboard (shared state), Bulletin (agent communication), and AsyncDispatcher (async job handling) in real-world scenarios.

## Blackboard: Shared State Management

The Blackboard provides thread-safe shared state storage with automatic expiration and reactive callbacks. It's ideal for caching, configuration sharing, and event-driven programming.

### Basic Usage

```python
from dachi.comm import Blackboard

# Create a blackboard instance
bb = Blackboard()

# Store data
bb.set("config", {"model": "gpt-4", "temperature": 0.7})
bb.set("user_session", "user123", scope="session_001")

# Retrieve data
config = bb.get("config")  # Returns dict
session = bb.get("user_session", scope="session_001")  # Returns "user123"

# Check existence
if bb.has("config"):
    print("Configuration is available")
```

### TTL (Time-To-Live) Pattern

```python
import time

# Store data with automatic expiration
bb.set_with_ttl("temp_token", "abc123", ttl_seconds=300)  # Expires in 5 minutes
bb.set_with_ttl("cache_result", {"data": [1, 2, 3]}, ttl_seconds=60)

# Check if still available
if bb.has("temp_token"):
    token = bb.get("temp_token")
    print(f"Token still valid: {token}")
else:
    print("Token has expired")

# Wait and check again
time.sleep(61)
cache = bb.get("cache_result")  # Returns None (expired)
```

### Reactive Pattern with Callbacks

```python
from dachi.comm import BlackboardEventType

def on_config_change(key: str, value: any, event_type: BlackboardEventType):
    print(f"Configuration changed: {key} = {value}")

def on_cache_expire(key: str, value: any, event_type: BlackboardEventType):
    print(f"Cache expired: {key}")

# Register callbacks
bb.register_callback("config", on_config_change, events={BlackboardEventType.ON_SET})
bb.register_callback("cache_*", on_cache_expire, events={BlackboardEventType.ON_EXPIRE})

# Triggers callback
bb.set("config", {"model": "gpt-3.5-turbo"})

# Will trigger expiration callback after TTL
bb.set_with_ttl("cache_user_data", {"id": 123}, ttl_seconds=1)
time.sleep(2)  # Callback fires when data expires
```

### Multi-Scope Pattern

```python
# Use scopes to isolate data between different contexts
bb.set("current_task", "analyze_data", scope="agent_001")
bb.set("current_task", "generate_report", scope="agent_002")
bb.set("shared_config", {"debug": True})  # No scope = global

# Each agent sees only its scoped data
task_001 = bb.get("current_task", scope="agent_001")  # "analyze_data"
task_002 = bb.get("current_task", scope="agent_002")  # "generate_report"
config = bb.get("shared_config")  # Available to all agents

# Get statistics per scope
stats = bb.get_stats(scope="agent_001")
print(f"Agent 001 has {stats['count']} items stored")
```

## Bulletin: Agent Communication

The Bulletin provides a message board for posting and retrieving typed messages between agents or tasks. It supports filtering, locking, and scoped communication.

### Basic Message Passing

```python
from pydantic import BaseModel
from dachi.comm import Bulletin

class TaskRequest(BaseModel):
    task_id: str
    action: str
    params: dict

class TaskResult(BaseModel):
    task_id: str
    status: str
    result: any

# Create bulletins for different message types
requests = Bulletin[TaskRequest]()
results = Bulletin[TaskResult]()

# Agent A posts a request
request = TaskRequest(
    task_id="task_001",
    action="process_data",
    params={"file": "data.csv", "method": "analyze"}
)
requests.publish(request)

# Agent B retrieves and processes the request
pending_request = requests.retrieve_first(filters={"action": "process_data"})
if pending_request:
    # Process the task...
    result = TaskResult(
        task_id=pending_request.task_id,
        status="completed",
        result={"summary": "Data analyzed successfully"}
    )
    results.publish(result)
```

### Work Queue Pattern

```python
from typing import List

# Worker pulls tasks from a queue
def worker_loop(worker_id: str):
    while True:
        # Retrieve and lock a task
        task = requests.retrieve_first(
            filters={"status": "pending"},
            lock_duration_seconds=60  # Lock for 1 minute
        )
        
        if task is None:
            time.sleep(1)  # No tasks available
            continue
        
        try:
            # Process the task
            print(f"Worker {worker_id} processing {task.task_id}")
            time.sleep(5)  # Simulate work
            
            # Mark as completed and release
            results.publish(TaskResult(
                task_id=task.task_id,
                status="completed",
                result={"processed_by": worker_id}
            ))
            requests.remove(task.task_id)
            
        except Exception as e:
            # Release the lock so another worker can try
            requests.release(task.task_id)
            print(f"Task {task.task_id} failed: {e}")

# Start multiple workers
import threading
for i in range(3):
    thread = threading.Thread(target=worker_loop, args=[f"worker_{i}"])
    thread.daemon = True
    thread.start()
```

### Event-Driven Communication

```python
from dachi.comm import BulletinEventType

def on_new_request(item: TaskRequest, event_type: BulletinEventType):
    print(f"New task request: {item.task_id}")

def on_task_complete(item: TaskResult, event_type: BulletinEventType):
    if item.status == "completed":
        print(f"Task {item.task_id} completed successfully")

# Register event handlers
requests.register_callback(on_new_request, events={BulletinEventType.ON_PUBLISH})
results.register_callback(on_task_complete, events={BulletinEventType.ON_PUBLISH})

# Events will fire when messages are published
requests.publish(TaskRequest(task_id="task_002", action="backup", params={}))
```

### Scoped Communication

```python
# Use scopes to prevent cross-agent interference
agent_001_requests = Bulletin[TaskRequest](scope="agent_001")
agent_002_requests = Bulletin[TaskRequest](scope="agent_002")
global_announcements = Bulletin[dict]()  # No scope = global

# Each agent only sees its own requests
agent_001_requests.publish(TaskRequest(task_id="a1_task", action="analyze", params={}))
agent_002_requests.publish(TaskRequest(task_id="a2_task", action="report", params={}))

# Agents retrieve only their scoped messages
a1_tasks = agent_001_requests.retrieve_all()  # Only a1_task
a2_tasks = agent_002_requests.retrieve_all()  # Only a2_task

# Global announcements visible to all
global_announcements.publish({"type": "system_maintenance", "message": "System will restart in 5 minutes"})
```

## AsyncDispatcher: Async Job Handling

The AsyncDispatcher coordinates asynchronous AI requests while maintaining thread safety for synchronous behavior trees. It manages job queues, concurrency limits, and provides both polling and callback interfaces.

### Basic Request Submission

```python
import asyncio
from dachi.comm import AsyncDispatcher
from dachi.proc import OpenAIChat
from dachi.utils import Msg

# Create dispatcher with concurrency limit
dispatcher = AsyncDispatcher(max_concurrent_requests=5)

# Create an AI processor
openai_proc = OpenAIChat(
    model="gpt-4",
    temperature=0.7
)

# Submit a request
message = Msg(content="Explain quantum computing in simple terms")
request_id = dispatcher.submit_proc(openai_proc, message, callback_id="explain_quantum")

# Poll for completion
import time
while True:
    status = dispatcher.status(request_id)
    if status.is_complete():
        result = dispatcher.result(request_id)
        print(f"AI Response: {result.content}")
        break
    time.sleep(0.1)
```

### Callback Pattern

```python
from dachi.comm import RequestStatus

def on_completion(request_id: str, result: any, status: RequestStatus):
    if status == RequestStatus.DONE:
        print(f"Request {request_id} completed: {result.content}")
    elif status == RequestStatus.ERROR:
        print(f"Request {request_id} failed: {result}")

# Submit request with callback
request_id = dispatcher.submit_proc(
    openai_proc, 
    message, 
    callback=on_completion,
    callback_id="async_task"
)

# Callback will be invoked when complete
# Continue with other work...
```

### Streaming Pattern

```python
def handle_stream_chunk(request_id: str, chunk: str):
    print(f"Stream chunk: {chunk}", end="")

def handle_stream_complete(request_id: str, final_result: str):
    print(f"\nStream completed for {request_id}")

# Submit streaming request
request_id = dispatcher.submit_stream(
    openai_proc,
    message,
    chunk_callback=handle_stream_chunk,
    completion_callback=handle_stream_complete
)

# Stream results are delivered via callbacks
time.sleep(5)  # Let stream complete
```

### Batch Processing Pattern

```python
# Submit multiple requests for parallel processing
messages = [
    Msg(content="Summarize the benefits of renewable energy"),
    Msg(content="Explain the water cycle"),
    Msg(content="Describe photosynthesis process"),
    Msg(content="What causes earthquakes?")
]

request_ids = []
for i, msg in enumerate(messages):
    req_id = dispatcher.submit_proc(openai_proc, msg, callback_id=f"batch_{i}")
    request_ids.append(req_id)

# Wait for all to complete
completed_results = []
while len(completed_results) < len(request_ids):
    for req_id in request_ids:
        if req_id not in [r[0] for r in completed_results]:
            status = dispatcher.status(req_id)
            if status.is_complete():
                result = dispatcher.result(req_id)
                completed_results.append((req_id, result))
    time.sleep(0.1)

print(f"Completed {len(completed_results)} requests")
```

### Integration with Behavior Trees

```python
from dachi.act import Task, TaskStatus

class AIRequestTask(Task):
    def __init__(self, dispatcher: AsyncDispatcher, processor, message: Msg):
        super().__init__()
        self.dispatcher = dispatcher
        self.processor = processor
        self.message = message
        self.request_id = None
    
    def tick(self) -> TaskStatus:
        if self.request_id is None:
            # Submit request on first tick
            self.request_id = self.dispatcher.submit_proc(
                self.processor, 
                self.message
            )
            return TaskStatus.RUNNING
        
        # Check status on subsequent ticks
        status = self.dispatcher.status(self.request_id)
        if status == RequestStatus.DONE:
            self.result = self.dispatcher.result(self.request_id)
            return TaskStatus.SUCCESS
        elif status == RequestStatus.ERROR:
            return TaskStatus.FAILURE
        else:
            return TaskStatus.RUNNING

# Use in behavior tree
ai_task = AIRequestTask(dispatcher, openai_proc, Msg(content="Hello, AI!"))
result = ai_task.tick()  # Returns RUNNING initially
# ... continue ticking until SUCCESS or FAILURE
```

## Best Practices

### Thread Safety
- All three components (Blackboard, Bulletin, AsyncDispatcher) are thread-safe
- Use scopes to isolate data between different agents/contexts
- Register callbacks before starting concurrent operations

### Error Handling
- Always check for None returns from `get()` and `retrieve_first()`
- Use try/catch blocks when processing retrieved data
- Monitor RequestStatus for ERROR states in AsyncDispatcher

### Performance
- Use TTL to prevent memory leaks in long-running systems
- Set appropriate concurrency limits in AsyncDispatcher
- Use filters in Bulletin to avoid retrieving irrelevant messages

### Memory Management
- Call `clear()` periodically to clean up expired data
- Use `remove()` instead of `release()` for completed work items
- Monitor memory usage with `get_stats()` in production