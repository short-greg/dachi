# Quick Start Guide

Get up and running with Dachi in under 10 minutes. This guide shows you how to build a simple AI agent that can process requests and maintain conversation state.

## Prerequisites

```bash
# Install Dachi (assuming it's available via pip)
pip install dachi

# Or if working from source
pip install -e .

# You'll also need an OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Hello World: Basic AI Agent

Create a file called `hello_dachi.py`:

```python
from dachi.comm import Blackboard, AsyncDispatcher
from dachi.proc import OpenAIChat
from dachi.utils import Msg

# Create the core components
blackboard = Blackboard()
dispatcher = AsyncDispatcher(max_concurrent_requests=2)
ai_processor = OpenAIChat(model="gpt-4", temperature=0.7)

# Initialize conversation state
blackboard.set("messages", [])
blackboard.set("conversation_count", 0)

def chat(user_message: str) -> str:
    """Send a message to the AI and get a response"""
    
    # Update conversation history
    messages = blackboard.get("messages", [])
    messages.append({"role": "user", "content": user_message})
    blackboard.set("messages", messages)
    
    # Create AI request
    ai_message = Msg(content=user_message)
    
    # Submit async request
    request_id = dispatcher.submit_proc(ai_processor, ai_message)
    
    # Wait for response
    import time
    while True:
        status = dispatcher.status(request_id)
        if status.is_complete():
            result = dispatcher.result(request_id)
            
            # Update conversation history
            messages.append({"role": "assistant", "content": result.content})
            blackboard.set("messages", messages)
            
            # Update stats
            count = blackboard.get("conversation_count", 0) + 1
            blackboard.set("conversation_count", count)
            
            return result.content
        
        time.sleep(0.1)

# Test it out
if __name__ == "__main__":
    print("Dachi Hello World - Simple AI Chat")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input:
            response = chat(user_input)
            print(f"AI: {response}\n")
            
            # Show conversation stats
            count = blackboard.get("conversation_count")
            print(f"[Conversation turns: {count}]")
```

Run it:

```bash
python hello_dachi.py
```

## Example Interaction

```
Dachi Hello World - Simple AI Chat
Type 'quit' to exit

You: Hello! What is Dachi?
AI: Hello! Dachi is an AI framework for building intelligent systems using Large Language Models (LLMs). It provides flexible interaction with LLMs, task coordination through behavior trees, and customizable workflows for AI agents. The framework includes components for communication, state management, and async processing to help developers build sophisticated AI applications.

[Conversation turns: 1]

You: Show me a simple behavior tree example
AI: Here's a simple behavior tree example using Dachi:

```python
from dachi.act import Task, TaskStatus, Sequence

class CheckWeatherTask(Task):
    def tick(self):
        # Simulate checking weather
        print("Checking weather...")
        return TaskStatus.SUCCESS

class DecideActivityTask(Task): 
    def tick(self):
        print("Deciding on activity based on weather...")
        return TaskStatus.SUCCESS

# Create a sequence that runs tasks in order
morning_routine = Sequence("morning_routine")
morning_routine.add_child(CheckWeatherTask("check_weather"))
morning_routine.add_child(DecideActivityTask("decide_activity"))

# Execute the behavior tree
status = morning_routine.tick()  # Returns SUCCESS if all tasks succeed
```

This creates a behavior tree that checks weather first, then decides on an activity - only proceeding to the second task if the first succeeds.

[Conversation turns: 2]
```

## 5-Minute Tutorial: Smart Task Processor

Let's build something more interesting - a task processor that can handle different types of work:

```python
# smart_processor.py
from dachi.comm import Blackboard, Bulletin, AsyncDispatcher
from dachi.proc import OpenAIChat
from dachi.utils import Msg
from pydantic import BaseModel
from typing import Dict, Any
import uuid

class TaskRequest(BaseModel):
    task_id: str
    task_type: str  # "analyze", "summarize", "translate"
    data: Dict[str, Any]

class TaskResult(BaseModel):
    task_id: str
    result: str
    processing_time: float

class SmartProcessor:
    def __init__(self):
        # Core components
        self.blackboard = Blackboard()
        self.task_queue = Bulletin[TaskRequest]()
        self.results = Bulletin[TaskResult]()
        self.dispatcher = AsyncDispatcher(max_concurrent_requests=3)
        self.ai_processor = OpenAIChat(model="gpt-4", temperature=0.3)
        
        # Initialize stats
        self.blackboard.set("tasks_completed", 0)
        self.blackboard.set("tasks_failed", 0)
    
    def submit_task(self, task_type: str, data: Dict[str, Any]) -> str:
        """Submit a task for processing"""
        task_id = str(uuid.uuid4())[:8]
        
        task = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            data=data
        )
        
        self.task_queue.publish(task)
        print(f"Submitted {task_type} task: {task_id}")
        return task_id
    
    def process_next_task(self) -> bool:
        """Process the next available task"""
        import time
        
        # Get next task
        task = self.task_queue.retrieve_first()
        if not task:
            return False
        
        print(f"Processing task {task.task_id} ({task.task_type})")
        start_time = time.time()
        
        try:
            # Create AI prompt based on task type
            if task.task_type == "analyze":
                prompt = f"Analyze the following text and provide key insights: {task.data.get('text', '')}"
            elif task.task_type == "summarize":
                prompt = f"Summarize the following text in 2-3 sentences: {task.data.get('text', '')}"
            elif task.task_type == "translate":
                target_lang = task.data.get('target_language', 'Spanish')
                prompt = f"Translate the following text to {target_lang}: {task.data.get('text', '')}"
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Submit to AI
            ai_message = Msg(content=prompt)
            request_id = self.dispatcher.submit_proc(self.ai_processor, ai_message)
            
            # Wait for result
            while True:
                status = self.dispatcher.status(request_id)
                if status.is_complete():
                    ai_result = self.dispatcher.result(request_id)
                    break
                time.sleep(0.1)
            
            # Create and publish result
            processing_time = time.time() - start_time
            result = TaskResult(
                task_id=task.task_id,
                result=ai_result.content,
                processing_time=processing_time
            )
            
            self.results.publish(result)
            self.task_queue.remove(task.task_id)
            
            # Update stats
            completed = self.blackboard.get("tasks_completed", 0) + 1
            self.blackboard.set("tasks_completed", completed)
            
            print(f"✓ Completed task {task.task_id} in {processing_time:.2f}s")
            return True
            
        except Exception as e:
            # Handle error
            failed = self.blackboard.get("tasks_failed", 0) + 1
            self.blackboard.set("tasks_failed", failed)
            
            self.task_queue.release(task.task_id)  # Release for retry
            print(f"✗ Task {task.task_id} failed: {e}")
            return False
    
    def get_result(self, task_id: str, timeout: float = 30.0) -> str:
        """Get result for a specific task"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.results.retrieve_first(filters={"task_id": task_id})
            if result:
                return result.result
            time.sleep(0.5)
        
        return "Task result not available (timeout or still processing)"
    
    def worker_loop(self):
        """Run continuous processing loop"""
        import time
        print("Worker started - processing tasks...")
        
        while True:
            try:
                if not self.process_next_task():
                    time.sleep(1)  # No tasks available
            except KeyboardInterrupt:
                print("Worker stopped")
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "tasks_completed": self.blackboard.get("tasks_completed", 0),
            "tasks_failed": self.blackboard.get("tasks_failed", 0),
            "pending_tasks": len(self.task_queue.retrieve_all()),
            "available_results": len(self.results.retrieve_all())
        }

# Demo usage
def main():
    processor = SmartProcessor()
    
    # Submit some tasks
    analyze_id = processor.submit_task("analyze", {
        "text": "The rise of artificial intelligence is transforming industries worldwide. From healthcare to finance, AI is enabling new capabilities and efficiencies."
    })
    
    summarize_id = processor.submit_task("summarize", {
        "text": "Machine learning algorithms can identify patterns in large datasets that humans might miss. This capability is particularly valuable in medical diagnosis, where ML can analyze medical images and patient data to suggest potential diagnoses or treatments."
    })
    
    translate_id = processor.submit_task("translate", {
        "text": "Hello, world! Welcome to the future of AI.",
        "target_language": "French"
    })
    
    # Process tasks
    import threading
    worker_thread = threading.Thread(target=processor.worker_loop, daemon=True)
    worker_thread.start()
    
    # Wait and display results
    import time
    print("\nWaiting for results...\n")
    time.sleep(10)  # Give tasks time to process
    
    print("RESULTS:")
    print("="*50)
    
    print(f"Analysis: {processor.get_result(analyze_id)}")
    print("\n" + "-"*50)
    
    print(f"Summary: {processor.get_result(summarize_id)}")
    print("\n" + "-"*50)
    
    print(f"Translation: {processor.get_result(translate_id)}")
    print("\n" + "-"*50)
    
    print(f"Stats: {processor.get_stats()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python smart_processor.py
```

## What Just Happened?

In just a few lines of code, you built a sophisticated AI system with:

1. **Async Processing**: AI requests run asynchronously without blocking
2. **Message Queues**: Tasks are queued and processed efficiently  
3. **Shared State**: Statistics and results are stored in thread-safe shared memory
4. **Type Safety**: Pydantic models ensure message structure
5. **Error Handling**: Failed tasks can be retried

## Core Concepts Demonstrated

### Blackboard (Shared State)
```python
blackboard.set("key", value)  # Store data
value = blackboard.get("key")  # Retrieve data
```

### Bulletin (Message Passing)
```python
bulletin.publish(message)       # Send message
message = bulletin.retrieve_first()  # Get message
```

### AsyncDispatcher (AI Processing)
```python
request_id = dispatcher.submit_proc(ai_processor, message)
result = dispatcher.result(request_id)  # Get result when ready
```

## Next Steps

1. **Read the tutorials**:
   - [Simple Chat Agent](tutorial-simple-chat-agent.md) - Build a conversational AI
   - [Multi-Agent Communication](tutorial-multi-agent-communication.md) - Coordinate multiple agents

2. **Explore usage patterns**:
   - [Usage Patterns](usage-patterns.md) - Canonical patterns for each component
   - [Architecture in Practice](architecture-in-practice.md) - How components work together

3. **Try behavior trees**:
   ```python
   from dachi.act import Task, TaskStatus, Sequence, Parallel
   # Build complex decision trees and state machines
   ```

4. **Add streaming**:
   ```python
   # For real-time AI responses
   dispatcher.submit_stream(ai_processor, message, chunk_callback=handle_chunk)
   ```

## Common Patterns

### Request-Response with State
```python
# Store request context
blackboard.set("current_user", user_id)
blackboard.set("conversation_context", context)

# Process with AI
request_id = dispatcher.submit_proc(ai_processor, message)
response = wait_for_result(request_id)

# Update state with result
update_conversation_history(response)
```

### Multi-Step Workflows
```python
# Task 1: Analyze
analyze_task = submit_analysis_task(data)

# Task 2: Process (depends on analysis)
process_task = submit_processing_task(depends_on=[analyze_task])

# Task 3: Report (depends on processing)  
report_task = submit_report_task(depends_on=[process_task])
```

### Event-Driven Coordination
```python
def on_task_complete(result, event_type):
    if result.task_type == "analysis":
        # Trigger next step
        submit_followup_task(result.data)

bulletin.register_callback(on_task_complete, events={BulletinEventType.ON_PUBLISH})
```

You're now ready to build sophisticated AI applications with Dachi! 🚀