# Architecture in Practice

This guide demonstrates how Dachi's core components work together in real-world scenarios. We'll trace data flow through the system and show how messaging, state management, and async dispatching integrate with behavior trees and state machines.

## System Architecture Overview

Dachi's architecture consists of four main layers that work together:

1. **Foundation Layer**: BaseModule, Process, Task, ShareableItem hierarchy
2. **Communication Layer**: Msg/Resp, Bulletin, Blackboard, AsyncDispatcher  
3. **AI Integration Layer**: OpenAI adapters, custom processors
4. **Control Flow Layer**: Behavior trees, state machines, async coordination

## Data Flow Patterns

### Pattern 1: Request → Processing → State Update

This is the most common pattern in AI applications: receive a request, process it asynchronously, and update shared state.

```python
from dachi.comm import Blackboard, AsyncDispatcher
from dachi.proc import OpenAIChat
from dachi.utils import Msg
from dachi.act import Task, TaskStatus

class ChatBot:
    def __init__(self):
        # Initialize communication components
        self.blackboard = Blackboard()
        self.dispatcher = AsyncDispatcher(max_concurrent_requests=3)
        self.ai_processor = OpenAIChat(model="gpt-4", temperature=0.7)
        
        # Initialize conversation state
        self.blackboard.set("conversation_history", [])
        self.blackboard.set("bot_status", "ready")
    
    def process_user_message(self, user_message: str) -> str:
        """Process user message and return bot response"""
        
        # 1. Update state with user message
        history = self.blackboard.get("conversation_history", [])
        history.append({"role": "user", "content": user_message})
        self.blackboard.set("conversation_history", history)
        self.blackboard.set("bot_status", "processing")
        
        # 2. Create AI request with context
        context = self._build_context(history)
        message = Msg(content=user_message, context=context)
        
        # 3. Submit async request
        request_id = self.dispatcher.submit_proc(
            self.ai_processor, 
            message,
            callback_id=f"chat_{len(history)}"
        )
        
        # 4. Wait for completion (in real app, this would be non-blocking)
        result = self._wait_for_result(request_id)
        
        # 5. Update state with response
        history.append({"role": "assistant", "content": result.content})
        self.blackboard.set("conversation_history", history)
        self.blackboard.set("bot_status", "ready")
        
        return result.content
    
    def _build_context(self, history: list) -> dict:
        return {
            "conversation_length": len(history),
            "recent_topics": self._extract_topics(history[-5:]),
            "user_sentiment": self._analyze_sentiment(history)
        }
    
    def _wait_for_result(self, request_id: str):
        """Synchronous wait - use callbacks in real applications"""
        import time
        while True:
            status = self.dispatcher.status(request_id)
            if status.is_complete():
                return self.dispatcher.result(request_id)
            time.sleep(0.1)

# Usage
bot = ChatBot()
response = bot.process_user_message("Tell me about quantum computing")
print(response)
```

### Pattern 2: Multi-Agent Task Distribution

Shows how multiple agents coordinate using Bulletin for task distribution and Blackboard for shared state.

```python
from pydantic import BaseModel
from dachi.comm import Bulletin, Blackboard
from threading import Thread
import time
import uuid

class WorkItem(BaseModel):
    id: str
    task_type: str
    data: dict
    priority: int = 1

class WorkResult(BaseModel):
    work_id: str
    result: dict
    processing_time: float
    worker_id: str

class MultiAgentSystem:
    def __init__(self):
        # Shared communication channels
        self.work_queue = Bulletin[WorkItem]()
        self.results = Bulletin[WorkResult]()
        self.blackboard = Blackboard()
        
        # System state
        self.blackboard.set("active_workers", set())
        self.blackboard.set("work_stats", {"submitted": 0, "completed": 0, "failed": 0})
        
        # Start workers
        self.workers = []
        for i in range(3):
            worker = WorkerAgent(f"worker_{i}", self)
            self.workers.append(worker)
            worker.start()
    
    def submit_work(self, task_type: str, data: dict, priority: int = 1) -> str:
        """Submit work to the distributed system"""
        work_id = str(uuid.uuid4())
        work_item = WorkItem(
            id=work_id,
            task_type=task_type,
            data=data,
            priority=priority
        )
        
        # Publish to work queue
        self.work_queue.publish(work_item)
        
        # Update stats
        stats = self.blackboard.get("work_stats")
        stats["submitted"] += 1
        self.blackboard.set("work_stats", stats)
        
        return work_id
    
    def get_result(self, work_id: str, timeout: float = 30.0) -> WorkResult:
        """Wait for and retrieve work result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.results.retrieve_first(filters={"work_id": work_id})
            if result:
                return result
            time.sleep(0.1)
        raise TimeoutError(f"Work {work_id} did not complete within {timeout}s")
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            "active_workers": len(self.blackboard.get("active_workers", set())),
            "pending_work": len(self.work_queue.retrieve_all()),
            "stats": self.blackboard.get("work_stats")
        }

class WorkerAgent:
    def __init__(self, worker_id: str, system: MultiAgentSystem):
        self.worker_id = worker_id
        self.system = system
        self.thread = None
        self.running = False
    
    def start(self):
        """Start the worker thread"""
        self.running = True
        self.thread = Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Register with system
        active_workers = self.system.blackboard.get("active_workers", set())
        active_workers.add(self.worker_id)
        self.system.blackboard.set("active_workers", active_workers)
    
    def _worker_loop(self):
        """Main worker processing loop"""
        while self.running:
            try:
                # Get work item with lock
                work = self.system.work_queue.retrieve_first(
                    lock_duration_seconds=60,
                    filters={}  # Accept any work
                )
                
                if work is None:
                    time.sleep(1)  # No work available
                    continue
                
                # Process the work
                start_time = time.time()
                result_data = self._process_work(work)
                processing_time = time.time() - start_time
                
                # Create result
                result = WorkResult(
                    work_id=work.id,
                    result=result_data,
                    processing_time=processing_time,
                    worker_id=self.worker_id
                )
                
                # Publish result and remove work item
                self.system.results.publish(result)
                self.system.work_queue.remove(work.id)
                
                # Update stats
                stats = self.system.blackboard.get("work_stats")
                stats["completed"] += 1
                self.system.blackboard.set("work_stats", stats)
                
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                # Release work item so another worker can try
                if 'work' in locals():
                    self.system.work_queue.release(work.id)
                
                stats = self.system.blackboard.get("work_stats")
                stats["failed"] += 1
                self.system.blackboard.set("work_stats", stats)
    
    def _process_work(self, work: WorkItem) -> dict:
        """Process work item - customize based on task_type"""
        if work.task_type == "analyze_text":
            # Simulate text analysis
            time.sleep(2)
            return {
                "word_count": len(work.data.get("text", "").split()),
                "analyzed_by": self.worker_id
            }
        elif work.task_type == "generate_summary":
            # Simulate summary generation
            time.sleep(3)
            return {
                "summary": f"Summary of {work.data.get('title', 'document')}",
                "generated_by": self.worker_id
            }
        else:
            return {"error": f"Unknown task type: {work.task_type}"}

# Usage example
system = MultiAgentSystem()

# Submit work
work_id1 = system.submit_work("analyze_text", {"text": "This is sample text to analyze"})
work_id2 = system.submit_work("generate_summary", {"title": "Research Paper", "content": "..."})

# Get results
result1 = system.get_result(work_id1)
result2 = system.get_result(work_id2)

print(f"Analysis result: {result1.result}")
print(f"Summary result: {result2.result}")
print(f"System status: {system.get_system_status()}")
```

### Pattern 3: Behavior Tree Integration

Shows how behavior trees integrate with async processing and shared state.

```python
from dachi.act import Task, TaskStatus, Sequence, Parallel
from dachi.comm import AsyncDispatcher, Blackboard
from dachi.proc import OpenAIChat
from dachi.utils import Msg

class AIAnalysisTask(Task):
    """Behavior tree task that performs AI analysis"""
    
    def __init__(self, name: str, dispatcher: AsyncDispatcher, processor: OpenAIChat, 
                 blackboard: Blackboard, query: str):
        super().__init__(name=name)
        self.dispatcher = dispatcher
        self.processor = processor
        self.blackboard = blackboard
        self.query = query
        self.request_id = None
    
    def tick(self) -> TaskStatus:
        if self.request_id is None:
            # First tick: submit async request
            message = Msg(content=self.query)
            self.request_id = self.dispatcher.submit_proc(
                self.processor, 
                message,
                callback_id=f"analysis_{self.name}"
            )
            return TaskStatus.RUNNING
        
        # Subsequent ticks: check status
        status = self.dispatcher.status(self.request_id)
        if status.is_complete():
            if status.is_success():
                # Store result in blackboard
                result = self.dispatcher.result(self.request_id)
                self.blackboard.set(f"analysis_{self.name}", result.content)
                return TaskStatus.SUCCESS
            else:
                # Store error
                error = self.dispatcher.result(self.request_id)
                self.blackboard.set(f"error_{self.name}", str(error))
                return TaskStatus.FAILURE
        
        return TaskStatus.RUNNING

class StateUpdateTask(Task):
    """Updates shared state based on analysis results"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        super().__init__(name=name)
        self.blackboard = blackboard
    
    def tick(self) -> TaskStatus:
        # Check if required analyses are complete
        sentiment = self.blackboard.get("analysis_sentiment")
        topics = self.blackboard.get("analysis_topics")
        summary = self.blackboard.get("analysis_summary")
        
        if sentiment and topics and summary:
            # Combine results
            final_result = {
                "sentiment": sentiment,
                "topics": topics,
                "summary": summary,
                "timestamp": time.time()
            }
            self.blackboard.set("final_analysis", final_result)
            return TaskStatus.SUCCESS
        
        return TaskStatus.FAILURE

class ComprehensiveAnalysisBehaviorTree:
    """Behavior tree that performs comprehensive text analysis"""
    
    def __init__(self):
        # Initialize components
        self.blackboard = Blackboard()
        self.dispatcher = AsyncDispatcher(max_concurrent_requests=5)
        self.ai_processor = OpenAIChat(model="gpt-4", temperature=0.3)
        
        # Build behavior tree
        self.root = self._build_tree()
    
    def _build_tree(self):
        """Build the behavior tree structure"""
        
        # Parallel analysis tasks (run concurrently)
        parallel_analysis = Parallel("parallel_analysis", policy="succeed_on_all")
        parallel_analysis.add_child(
            AIAnalysisTask("sentiment", self.dispatcher, self.ai_processor, self.blackboard,
                          "Analyze the sentiment of this text")
        )
        parallel_analysis.add_child(
            AIAnalysisTask("topics", self.dispatcher, self.ai_processor, self.blackboard,
                          "Extract main topics from this text")
        )
        parallel_analysis.add_child(
            AIAnalysisTask("summary", self.dispatcher, self.ai_processor, self.blackboard,
                          "Create a concise summary of this text")
        )
        
        # Sequential execution: analysis then state update
        root = Sequence("comprehensive_analysis")
        root.add_child(parallel_analysis)
        root.add_child(StateUpdateTask("combine_results", self.blackboard))
        
        return root
    
    def analyze_text(self, text: str) -> dict:
        """Perform comprehensive analysis of text"""
        # Set input text in blackboard
        self.blackboard.set("input_text", text)
        
        # Execute behavior tree
        while True:
            status = self.root.tick()
            if status == TaskStatus.SUCCESS:
                return self.blackboard.get("final_analysis")
            elif status == TaskStatus.FAILURE:
                errors = {
                    k: v for k, v in self.blackboard.items() 
                    if k.startswith("error_")
                }
                raise Exception(f"Analysis failed: {errors}")
            
            time.sleep(0.1)  # Wait before next tick

# Usage
analyzer = ComprehensiveAnalysisBehaviorTree()
text = "Artificial intelligence is revolutionizing how we work and live..."
result = analyzer.analyze_text(text)
print(f"Analysis complete: {result}")
```

## Control Flow Patterns

### Async/Sync Coordination

The key challenge in AI applications is coordinating synchronous behavior trees with asynchronous AI calls. Dachi solves this with the AsyncDispatcher pattern:

```python
# Synchronous behavior tree ticks...
def behavior_tree_loop():
    while True:
        status = root_task.tick()  # Synchronous
        if status.is_complete():
            break
        time.sleep(0.1)  # Fixed tick rate

# ...coordinate with async AI processing
class SmartTask(Task):
    def tick(self) -> TaskStatus:
        if not self.request_submitted:
            # Submit async request
            self.request_id = dispatcher.submit_proc(ai_proc, message)
            self.request_submitted = True
            return TaskStatus.RUNNING
        
        # Check async status synchronously
        status = dispatcher.status(self.request_id)
        if status.is_complete():
            return TaskStatus.SUCCESS if status.is_success() else TaskStatus.FAILURE
        return TaskStatus.RUNNING
```

### State Machine Integration

Behavior trees and state machines can work together through shared state:

```python
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_INPUT = "waiting_for_input"
    ERROR = "error"

class StatefulAgent:
    def __init__(self):
        self.blackboard = Blackboard()
        self.blackboard.set("agent_state", AgentState.IDLE)
        self.behavior_tree = self._build_behavior_tree()
    
    def update(self):
        """Called each frame/tick"""
        current_state = self.blackboard.get("agent_state")
        
        # State-specific behavior tree execution
        if current_state == AgentState.IDLE:
            # Look for new work
            if self._has_pending_work():
                self.blackboard.set("agent_state", AgentState.PROCESSING)
        
        elif current_state == AgentState.PROCESSING:
            # Execute behavior tree
            status = self.behavior_tree.tick()
            if status == TaskStatus.SUCCESS:
                self.blackboard.set("agent_state", AgentState.IDLE)
            elif status == TaskStatus.FAILURE:
                self.blackboard.set("agent_state", AgentState.ERROR)
        
        elif current_state == AgentState.ERROR:
            # Handle error recovery
            self._handle_error_recovery()
```

## Concurrency and Thread Safety

### Best Practices

1. **Use Scopes**: Isolate data between different agents/contexts
2. **Atomic Operations**: Combine related state updates
3. **Event-Driven Updates**: Use callbacks to avoid polling
4. **Resource Management**: Set appropriate concurrency limits

```python
# Good: Atomic state update
def update_conversation_atomically(blackboard: Blackboard, user_msg: str, ai_response: str):
    history = blackboard.get("conversation", [])
    history.extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ai_response}
    ])
    blackboard.set("conversation", history)

# Good: Scoped isolation
user_1_board = Blackboard(scope="user_001")
user_2_board = Blackboard(scope="user_002")

# Good: Event-driven coordination
def on_work_complete(item: WorkResult, event_type):
    # Update metrics
    stats = blackboard.get("completion_stats", {"total": 0, "avg_time": 0})
    stats["total"] += 1
    stats["avg_time"] = (stats["avg_time"] * (stats["total"] - 1) + item.processing_time) / stats["total"]
    blackboard.set("completion_stats", stats)

results.register_callback(on_work_complete, events={BulletinEventType.ON_PUBLISH})
```

## Performance Considerations

### Memory Management
- Use TTL for temporary data
- Periodically clear completed work items
- Monitor blackboard statistics

### Concurrency Tuning
- Set AsyncDispatcher limits based on AI provider rate limits
- Use appropriate queue sizes for Bulletin
- Balance between responsiveness and resource usage

### Error Handling
- Always check for None returns
- Implement retry logic for failed AI requests
- Use circuit breaker patterns for external services

This architecture enables building sophisticated AI systems that are both performant and maintainable, with clear separation of concerns and robust error handling.