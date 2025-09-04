# Core Architecture

Dachi's architecture is built on a foundation of composable components that work together to enable intelligent systems. Understanding these core concepts is essential for building effective AI applications.

## Foundation Layer: BaseModule/Process/State/Task

### BaseModule: The Universal Building Block

Everything in Dachi inherits from `BaseModule`. This provides the fundamental capabilities that enable learning and serialization.

**Key Features:**
- **Registry-based**: All modules are automatically registered for reconstruction
- **Spec/State separation**: Clean division between configuration and runtime state
- **Serializable**: Can persist and restore complete system state

```python
from dachi.core import BaseModule, Param, Attr, Shared

class SmartProcessor(BaseModule):
    # Configuration parameters (serialized with spec)
    model_name: str = Param(default="gpt-4")
    temperature: float = Param(default=0.7)
    
    # Runtime attributes (serialized with state)
    success_count: int = Attr(default=0)
    learned_patterns: dict = Attr(default_factory=dict)
    
    # Shared state across instances
    global_knowledge: dict = Shared(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        # Module is automatically registered
        # Can be reconstructed from spec + state
```

**Spec/State Pattern:**
```python
# Create and use module
processor = SmartProcessor(model_name="gpt-4", temperature=0.8)
processor.success_count = 10  # Runtime state
processor.learned_patterns["greeting"] = "friendly_response"

# Serialize configuration and state separately
spec = processor.to_spec()  # {"model_name": "gpt-4", "temperature": 0.8}
state = processor.get_state()  # {"success_count": 10, "learned_patterns": {...}}

# Later: reconstruct with all learned knowledge
restored = SmartProcessor.from_spec(spec, state=state)
assert restored.success_count == 10  # State preserved
assert restored.learned_patterns["greeting"] == "friendly_response"
```

### Process: Execution Patterns

`Process` defines how work gets done in Dachi. All processing components inherit from this to provide consistent execution patterns.

**Four Execution Modes:**
```python
from dachi.proc import Process

class TextProcessor(Process):
    def forward(self, input: str) -> str:
        """Synchronous processing"""
        return self.process_text(input)
    
    async def aforward(self, input: str) -> str:
        """Asynchronous processing"""
        return await self.async_process_text(input)
    
    def stream(self, input: str) -> Iterator[str]:
        """Synchronous streaming"""
        for chunk in self.stream_process(input):
            yield chunk
    
    async def astream(self, input: str) -> AsyncIterator[str]:
        """Asynchronous streaming"""
        async for chunk in self.async_stream_process(input):
            yield chunk
```

**Consistent Interface:**
```python
# Same interface works across all execution types
processor = TextProcessor()

# Pick the pattern you need
result = processor.forward("Hello")  # Sync
result = await processor.aforward("Hello")  # Async

# Streaming works the same way
for chunk in processor.stream("Hello"):
    print(chunk, end="")
    
async for chunk in processor.astream("Hello"):
    print(chunk, end="")
```

### State: Persistent Knowledge

State management in Dachi enables systems to learn and remember. The `ShareableItem` hierarchy provides different lifecycle patterns:

**ShareableItem Types:**
```python
from dachi.core import Param, Attr, Shared

class LearningAgent(BaseModule):
    # Param: Configuration - set once, serialized with spec
    model_type: str = Param(default="gpt-4")
    max_tokens: int = Param(default=1000)
    
    # Attr: Instance state - changes during execution, serialized with state
    conversation_count: int = Attr(default=0)
    user_preferences: dict = Attr(default_factory=dict)
    
    # Shared: Global state - shared across all instances
    collective_knowledge: dict = Shared(default_factory=dict)
    
    def handle_conversation(self, user_input: str):
        self.conversation_count += 1  # Updates instance state
        
        # Learn from interaction
        self.user_preferences["last_topic"] = extract_topic(user_input)
        
        # Update global knowledge
        self.collective_knowledge[user_input] = self.generate_response(user_input)
```

**State Persistence:**
```python
# Multiple agents can share knowledge
agent1 = LearningAgent()
agent2 = LearningAgent()

# Agent1 learns something
agent1.collective_knowledge["python"] = "programming_language"

# Agent2 immediately has access to this knowledge
print(agent2.collective_knowledge["python"])  # "programming_language"

# Save and restore complete system state
state_snapshot = {
    "agent1": agent1.get_state(),
    "agent2": agent2.get_state(),
    "shared": LearningAgent.get_shared_state()
}

# Later: restore entire system
new_agent1 = LearningAgent.from_spec(agent1.to_spec(), state=state_snapshot["agent1"])
new_agent2 = LearningAgent.from_spec(agent2.to_spec(), state=state_snapshot["agent2"])
LearningAgent.set_shared_state(state_snapshot["shared"])
```

### Task: Behavior Tree Nodes

`Task` is the foundation of Dachi's behavior tree system. It provides intelligent control flow and decision-making capabilities.

**TaskStatus Semantics:**
```python
from dachi.act import Task, TaskStatus

class IntelligentTask(Task):
    def tick(self) -> TaskStatus:
        if self.not_ready():
            return TaskStatus.WAITING  # Not ready to execute
        
        if self.already_running():
            return TaskStatus.RUNNING  # Continue execution
        
        try:
            result = self.execute()
            return TaskStatus.SUCCESS if result else TaskStatus.FAILURE
        except Exception:
            return TaskStatus.FAILURE
```

**Composable Behavior:**
```python
from dachi.act import Parallel, Selector, Sequence

# Sophisticated control flow
intelligent_agent = Selector([
    # Try high-priority tasks first
    Sequence([
        CheckEmergency(),
        HandleEmergency()
    ]),
    
    # Normal operation
    Parallel([
        MonitorEnvironment(),
        ExecutePlannedActions(),
        LearnFromExperience()
    ]),
    
    # Fallback
    ExploreNewOptions()
])

# Execute behavior tree
while True:
    status = intelligent_agent.tick()
    if status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
        break
    elif status == TaskStatus.WAITING:
        await asyncio.sleep(0.1)
```

**Task Composition Patterns:**
```python
# Tasks can be combined with logical operators
class SmartDecision(Task):
    def __init__(self):
        self.condition_a = CheckConditionA()
        self.condition_b = CheckConditionB()
        
    def tick(self) -> TaskStatus:
        # Use TaskStatus logical operations
        combined_status = self.condition_a.tick() & self.condition_b.tick()
        
        if combined_status == TaskStatus.SUCCESS:
            return self.execute_action()
        else:
            return TaskStatus.FAILURE
```

## How It All Works Together

The foundation components work together to enable intelligent, learning systems:

```python
class IntelligentConversationAgent(BaseModule, Process, Task):
    """Combines all foundation patterns"""
    
    # Configuration
    llm_model: str = Param(default="gpt-4")
    
    # Learning state
    conversation_history: list = Attr(default_factory=list)
    user_personality: dict = Attr(default_factory=dict)
    
    # Shared knowledge
    conversation_patterns: dict = Shared(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self.llm = self._create_llm()
    
    # Process interface - how work gets done
    async def aforward(self, user_input: str) -> str:
        # Update state (learning)
        self.conversation_history.append(user_input)
        self._analyze_user_personality(user_input)
        
        # Generate response using current state
        response = await self.llm.aforward(self._build_context())
        
        # Learn from interaction
        self._update_conversation_patterns(user_input, response)
        
        return response.msg.text
    
    # Task interface - intelligent control flow  
    def tick(self) -> TaskStatus:
        if self._should_initiate_conversation():
            asyncio.create_task(self._proactive_conversation())
            return TaskStatus.SUCCESS
        return TaskStatus.WAITING
    
    # Learning methods
    def _analyze_user_personality(self, input: str):
        """Learn about user preferences"""
        # Updates self.user_personality (Attr)
    
    def _update_conversation_patterns(self, input: str, response: str):
        """Update global conversation knowledge"""  
        # Updates self.conversation_patterns (Shared)
```

This foundation enables:
- **Intelligent behavior** through Task coordination
- **Learning** through state persistence
- **Consistent execution** through Process patterns  
- **System evolution** through BaseModule serialization

Next, we'll explore how messages and responses build on this foundation to create the communication layer.