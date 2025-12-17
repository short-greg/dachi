# Architecture in Practice

This guide demonstrates how Dachi's core components work together in real-world scenarios. We'll trace data flow through the system and show how text parameter optimization, messaging, state management, and async dispatching integrate with behavior trees and computational graphs.

## System Architecture Overview

Dachi's architecture consists of four main layers that work together:

1. **Foundation Layer**: BaseModule, Process, Task, ShareableItem hierarchy
2. **Text Parameter Layer**: PrivateParam, ParamSet, LangOptim, LangCritic
3. **Communication Layer**: Blackboard, Bulletin, AsyncDispatcher, Buffer
4. **Control Flow Layer**: Behavior trees, DataFlow graphs, async coordination

## Data Flow Patterns

### Pattern 1: Text Parameter Optimization Pipeline

This pattern shows how to optimize text parameters in a module using evaluation feedback.

```python
from dachi.core import Module, Param, PrivateParam
from dachi.proc import LangOptim, LangCritic, Process
from dachi.inst import PassFailCriterion, BoolField, TextField
from dachi.act.comm import Blackboard
import typing as t

class SummarizationModule(Module):
    """Module with optimizable text parameters"""

    _system_prompt: Param[str] = PrivateParam(
        default="You are a helpful summarization assistant",
        description="System prompt defining behavior"
    )

    _instruction: Param[str] = PrivateParam(
        default="Summarize the following text concisely",
        description="Task instruction"
    )

    _constraints: Param[str] = PrivateParam(
        default="Focus on key facts",
        description="Summarization constraints"
    )

    llm: Process  # Your LangModel implementation

    def forward(self, text: str) -> str:
        """Generate summary using current text parameters"""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"{self._instruction}\n\nText: {text}\n\n{self._constraints}"}
        ]
        response, _, _ = self.llm.forward(messages)
        return response

class SummarizationOptimizer(LangOptim):
    """Optimizer for summarization parameters"""

    def objective(self) -> str:
        return "Optimize parameters to generate concise, accurate summaries"

    def constraints(self) -> str:
        return "Maintain clarity and preserve key information"

    def param_evaluations(self, evaluations):
        return "\n".join([
            f"Input: {e.input}\nOutput: {e.output}\nPassed: {e.passed}"
            for e in evaluations
        ])

# Complete optimization pipeline
class SummarizationPipeline:
    def __init__(self, llm):
        # Initialize module with text parameters
        self.module = SummarizationModule(llm=llm)

        # Extract parameters for optimization
        self.param_set = self.module.param_set()

        # Define evaluation criterion
        self.criterion = PassFailCriterion(
            name="quality",
            passed=BoolField(description="Summary captures key points"),
            passing_criteria=TextField(description="Quality explanation")
        )

        # Create critic for evaluation
        self.critic = LangCritic(
            criterion=self.criterion,
            evaluator=llm,
            prompt_template="Evaluate: {output}\nOriginal: {input}"
        )

        # Create optimizer
        self.optimizer = SummarizationOptimizer(
            llm=llm,
            params=self.param_set,
            criterion=self.criterion
        )

        # Shared state for tracking
        self.blackboard = Blackboard()
        self.blackboard.set("iterations", 0)
        self.blackboard.set("best_pass_rate", 0.0)

    def optimize(self, test_texts: list[str], target_pass_rate: float = 0.9):
        """Optimize text parameters using evaluation feedback"""

        for iteration in range(10):
            # 1. Generate outputs with current parameters
            outputs = [self.module.forward(text) for text in test_texts]

            # 2. Evaluate outputs
            evaluations = self.critic.batch_forward(outputs, test_texts)

            # 3. Calculate pass rate
            pass_rate = sum(1 for e in evaluations if e.passed) / len(evaluations)

            # 4. Update shared state
            self.blackboard.set("iterations", iteration + 1)
            if pass_rate > self.blackboard.get("best_pass_rate"):
                self.blackboard.set("best_pass_rate", pass_rate)
                self.blackboard.set("best_params", {
                    "system_prompt": self.module._system_prompt,
                    "instruction": self.module._instruction,
                    "constraints": self.module._constraints
                })

            print(f"Iteration {iteration}: pass_rate={pass_rate:.2f}")

            # 5. Check convergence
            if pass_rate >= target_pass_rate:
                print(f"Converged at iteration {iteration}")
                break

            # 6. Update text parameters via LLM
            self.optimizer.step(evaluations)

        return self.blackboard.get("best_params")

# Usage
llm = my_llm_adapter  # Your LangModel implementation
pipeline = SummarizationPipeline(llm)

test_texts = [
    "Long article about AI...",
    "Research paper on quantum computing...",
    "News article about climate change..."
]

best_params = pipeline.optimize(test_texts)
print(f"Optimized parameters: {best_params}")
```

### Pattern 2: Multi-Agent Task Distribution

Shows how multiple agents coordinate using Bulletin for task distribution and Blackboard for shared state.

```python
from pydantic import BaseModel
from dachi.act.comm import Bulletin, Blackboard
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

### Pattern 3: Computational Graph with DataFlow

Shows how to build computational graphs (DAGs) using DataFlow for complex processing pipelines.

```python
from dachi.proc import DataFlow, V, Process
from dachi.core import Module, Param, PrivateParam
from dachi.act.comm import Blackboard
import time

class TextPreprocessor(Process):
    """Preprocesses text input"""

    def forward(self, text: str) -> str:
        # Clean and normalize text
        return text.strip().lower()

class SentimentAnalyzer(Module):
    """Analyzes sentiment with optimizable parameters"""

    _analysis_prompt: Param[str] = PrivateParam(
        default="Analyze the sentiment of this text",
        description="Sentiment analysis instruction"
    )

    llm: Process

    def forward(self, text: str) -> str:
        messages = [{"role": "user", "content": f"{self._analysis_prompt}\n\nText: {text}"}]
        response, _, _ = self.llm.forward(messages)
        return response

class TopicExtractor(Module):
    """Extracts topics with optimizable parameters"""

    _extraction_prompt: Param[str] = PrivateParam(
        default="Extract main topics from this text",
        description="Topic extraction instruction"
    )

    llm: Process

    def forward(self, text: str) -> str:
        messages = [{"role": "user", "content": f"{self._extraction_prompt}\n\nText: {text}"}]
        response, _, _ = self.llm.forward(messages)
        return response

class ResultAggregator(Process):
    """Aggregates analysis results"""

    def forward(self, preprocessed: str, sentiment: str, topics: str) -> dict:
        return {
            "preprocessed_text": preprocessed,
            "sentiment": sentiment,
            "topics": topics,
            "timestamp": time.time()
        }

class ComprehensiveAnalysisGraph:
    """DataFlow graph for comprehensive text analysis"""

    def __init__(self, llm):
        # Initialize processors
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer(llm=llm)
        self.topic_extractor = TopicExtractor(llm=llm)
        self.aggregator = ResultAggregator()

        # Build computational graph (DAG)
        self.graph = DataFlow(
            # Input node
            V("input_text"),

            # Preprocessing
            V("preprocessed") << self.preprocessor(V("input_text")),

            # Parallel analysis (both depend on preprocessed)
            V("sentiment") << self.sentiment_analyzer(V("preprocessed")),
            V("topics") << self.topic_extractor(V("preprocessed")),

            # Aggregation (depends on all analyses)
            V("result") << self.aggregator(
                V("preprocessed"),
                V("sentiment"),
                V("topics")
            )
        )

        # Shared state for tracking
        self.blackboard = Blackboard()
        self.blackboard.set("analysis_count", 0)

    def analyze(self, text: str) -> dict:
        """Execute the analysis graph"""
        # Run the computational graph
        result = self.graph.forward(input_text=text)

        # Update state
        count = self.blackboard.get("analysis_count", 0)
        self.blackboard.set("analysis_count", count + 1)
        self.blackboard.set("last_analysis", result["result"])

        return result["result"]

    async def aanalyze(self, text: str) -> dict:
        """Async version of analysis"""
        # DataFlow supports async execution
        result = await self.graph.aforward(input_text=text)

        count = self.blackboard.get("analysis_count", 0)
        self.blackboard.set("analysis_count", count + 1)

        return result["result"]

# Usage
llm = my_llm_adapter  # Your LangModel implementation
analyzer = ComprehensiveAnalysisGraph(llm)

# Synchronous analysis
text = "Artificial intelligence is revolutionizing how we work and live..."
result = analyzer.analyze(text)
print(f"Analysis: {result}")

# Async analysis
import asyncio
result = asyncio.run(analyzer.aanalyze(text))
print(f"Async analysis: {result}")

# Check state
print(f"Total analyses: {analyzer.blackboard.get('analysis_count')}")
```

**Key Benefits of DataFlow**:
1. **Automatic dependency resolution** - nodes execute when inputs are ready
2. **Parallel execution** - independent nodes run concurrently in async mode
3. **Clean DAG structure** - explicit data flow, no hidden dependencies
4. **Optimizable modules** - each module can have text parameters
5. **Reusable components** - processors compose into different graphs

## Control Flow Patterns

### Behavior Trees with Text Parameter Optimization

Combining behavior trees with optimizable modules enables adaptive decision-making:

```python
from dachi.act import Task, TaskStatus, Sequence, Selector
from dachi.core import Module, Param, PrivateParam
from dachi.act.comm import Blackboard

class DecisionModule(Module):
    """Decision-making module with optimizable parameters"""

    _decision_criteria: Param[str] = PrivateParam(
        default="Evaluate if action is beneficial",
        description="Decision-making criteria"
    )

    llm: Process

    def forward(self, context: dict) -> bool:
        messages = [{
            "role": "user",
            "content": f"{self._decision_criteria}\n\nContext: {context}"
        }]
        response, _, _ = self.llm.forward(messages)
        return "yes" in response.lower()

class DecisionTask(Task):
    """Behavior tree task that uses optimizable decision module"""

    def __init__(self, name: str, decision_module: DecisionModule, blackboard: Blackboard):
        super().__init__(name=name)
        self.decision_module = decision_module
        self.blackboard = blackboard

    def tick(self) -> TaskStatus:
        # Get current context
        context = self.blackboard.get("current_context", {})

        # Make decision using optimizable parameters
        should_proceed = self.decision_module.forward(context)

        if should_proceed:
            self.blackboard.set(f"decision_{self.name}", "proceed")
            return TaskStatus.SUCCESS
        else:
            self.blackboard.set(f"decision_{self.name}", "skip")
            return TaskStatus.FAILURE

class ActionTask(Task):
    """Task that executes an action"""

    def __init__(self, name: str, action: str, blackboard: Blackboard):
        super().__init__(name=name)
        self.action = action
        self.blackboard = blackboard

    def tick(self) -> TaskStatus:
        # Execute action
        print(f"Executing: {self.action}")
        self.blackboard.set(f"action_{self.name}_executed", True)
        return TaskStatus.SUCCESS

# Build adaptive behavior tree
class AdaptiveBehaviorTree:
    def __init__(self, llm):
        self.blackboard = Blackboard()
        self.decision_module = DecisionModule(llm=llm)

        # Build tree: try high-priority actions if decision passes
        self.root = Selector([
            # High priority path
            Sequence([
                DecisionTask("urgent", self.decision_module, self.blackboard),
                ActionTask("urgent_action", "Handle urgent matter", self.blackboard)
            ]),

            # Normal priority path
            Sequence([
                DecisionTask("normal", self.decision_module, self.blackboard),
                ActionTask("normal_action", "Handle normal matter", self.blackboard)
            ]),

            # Fallback
            ActionTask("fallback", "Wait and observe", self.blackboard)
        ])

    def execute(self, context: dict) -> str:
        """Execute behavior tree with given context"""
        self.blackboard.set("current_context", context)

        status = self.root.tick()

        if status == TaskStatus.SUCCESS:
            return "Action completed"
        else:
            return "Fallback executed"

# Usage - can optimize decision_module parameters
llm = my_llm_adapter
tree = AdaptiveBehaviorTree(llm)

context = {"urgency": "high", "resources": "available"}
result = tree.execute(context)
print(result)

# Optimize decision parameters
param_set = tree.decision_module.param_set()
# ... use LangOptim to improve decision-making ...
```

### State Machine Integration

Behavior trees and state machines can work together through shared state:

```python
from enum import Enum
from dachi.act.comm import Blackboard
from dachi.act import Task, TaskStatus

class AgentState(Enum):
    IDLE = "idle"
    OPTIMIZING = "optimizing"
    PROCESSING = "processing"
    ERROR = "error"

class StatefulAgent:
    def __init__(self, llm):
        self.blackboard = Blackboard()
        self.blackboard.set("agent_state", AgentState.IDLE)

        # Module with optimizable parameters
        self.processor = SummarizationModule(llm=llm)
        self.behavior_tree = self._build_behavior_tree()

    def update(self):
        """Called each frame/tick"""
        current_state = self.blackboard.get("agent_state")

        # State-specific behavior tree execution
        if current_state == AgentState.IDLE:
            # Look for new work
            if self._has_pending_work():
                self.blackboard.set("agent_state", AgentState.PROCESSING)
            elif self._should_optimize():
                self.blackboard.set("agent_state", AgentState.OPTIMIZING)

        elif current_state == AgentState.OPTIMIZING:
            # Optimize text parameters
            param_set = self.processor.param_set()
            # Run optimization loop
            # ... LangOptim steps ...
            self.blackboard.set("agent_state", AgentState.IDLE)

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
from dachi.act.comm import Blackboard, Bulletin, BulletinEventType

# Good: Atomic state update
def update_optimization_state(blackboard: Blackboard, iteration: int, pass_rate: float):
    """Atomic update of optimization state"""
    history = blackboard.get("optimization_history", [])
    history.append({
        "iteration": iteration,
        "pass_rate": pass_rate,
        "timestamp": time.time()
    })
    blackboard.set("optimization_history", history)
    blackboard.set("current_pass_rate", pass_rate)

# Good: Scoped isolation for multi-user systems
user_1_board = Blackboard(scope="user_001")
user_2_board = Blackboard(scope="user_002")

# Each user has isolated text parameters
user_1_board.set("custom_params", {"system_prompt": "..."})
user_2_board.set("custom_params", {"system_prompt": "..."})

# Good: Event-driven coordination
def on_evaluation_complete(evaluation: dict, event_type):
    """Update metrics when evaluation completes"""
    stats = blackboard.get("eval_stats", {"total": 0, "passed": 0})
    stats["total"] += 1
    if evaluation.get("passed"):
        stats["passed"] += 1
    stats["pass_rate"] = stats["passed"] / stats["total"]
    blackboard.set("eval_stats", stats)

evaluations_bulletin = Bulletin[dict]()
evaluations_bulletin.register_callback(
    on_evaluation_complete,
    events={BulletinEventType.ON_PUBLISH}
)
```

## Performance Considerations

### Memory Management
- Use TTL for temporary data in Blackboard
- Periodically clear completed work items from Bulletin
- Monitor blackboard statistics

### Text Parameter Optimization
- **Batch evaluations**: Evaluate multiple outputs at once with `critic.batch_forward()`
- **Early stopping**: Stop optimization when target pass rate is reached
- **Parameter caching**: Save optimized parameters via `module.to_spec()`
- **Parallel optimization**: Optimize different modules independently

### Async Processing
- Use `aforward()` and `astream()` for non-blocking LLM calls
- DataFlow automatically parallelizes independent nodes in async mode
- AsyncDispatcher coordinates multiple concurrent operations

### Error Handling
- Always check for None returns from Bulletin operations
- Implement retry logic for failed LLM requests
- Use try/except around text parameter updates
- Log optimization failures for analysis

## Architecture Summary

Dachi's architecture enables building sophisticated AI systems through:

1. **Text Parameter Optimization** - Bayesian updating of prompts via LangOptim
2. **Structured Evaluation** - Type-safe feedback via LangCritic + ResponseSpec
3. **Computational Graphs** - Clean DAG composition with DataFlow
4. **Behavior Trees** - Intelligent control flow with Task hierarchy
5. **State Management** - Blackboard for shared state, Bulletin for messaging
6. **Consistent Execution** - Process interface (sync/async × regular/streaming)

These components work together to create systems that are:
- **Optimizable** - Text parameters improve through feedback
- **Composable** - Mix and match components
- **Maintainable** - Clear separation of concerns
- **Performant** - Async execution and parallel processing
- **Transparent** - Inspect parameters and state at any time

See also:
- **[Optimization Guide](optimization-guide.md)** - Complete LangOptim workflow
- **[Behavior Trees](behavior-trees-and-coordination.md)** - Task coordination patterns
- **[Computational Graphs](computational-graphs.md)** - DataFlow composition
- **[Communication](communication-and-requests.md)** - Blackboard and Bulletin details