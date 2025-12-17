# Core Architecture

Dachi is a machine learning framework that uses **text as parameters** instead of numerical parameters. This architecture enables building adaptive AI systems through behavior trees, computational graphs, and LLM-driven optimization of text parameters via Bayesian updating.

This document covers the core architectural components that make this possible.

## Text Parameters: Dachi's Core Innovation

Traditional ML frameworks optimize numerical parameters (weights, biases) via gradient descent. Dachi optimizes **text parameters** (prompts, instructions, descriptions) via Bayesian updating with LLMs.

### Text as Parameters

Text parameters are defined using `Param[str]` with `PrivateParam`:

```python
from dachi.core import Module, Param, PrivateParam

class SummarizationModule(Module):
    _system_prompt: Param[str] = PrivateParam(
        default="You are a concise summarization assistant",
        description="System prompt that sets the summarization style"
    )

    _instruction: Param[str] = PrivateParam(
        default="Summarize the following text in 2-3 sentences",
        description="Specific task instruction for summarization"
    )

    _constraints: Param[str] = PrivateParam(
        default="Focus on key facts and main ideas",
        description="Constraints to guide the summarization"
    )

    def forward(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"{self._instruction}\n\nText: {text}\n\n{self._constraints}"}
        ]
        response, _, _ = self.llm.forward(messages)
        return response
```

**Why `PrivateParam`?**
- Marks parameters as optimizable by LangOptim
- Includes description for the optimizer to understand parameter purpose
- Private naming convention (`_param`) prevents external access
- Can be extracted as a ParamSet for optimization

### ParamSet: Parameter Collections

ParamSet groups related text parameters for batch optimization:

```python
# Extract all text parameters from a module
param_set = summarization_module.param_set()

# ParamSet provides:
# 1. to_schema() - JSON schema for LLM structured output
schema = param_set.to_schema()

# 2. update() - Apply optimizer suggestions
new_values = {
    "_system_prompt": "You are an expert summarization assistant...",
    "_instruction": "Create a comprehensive summary...",
    "_constraints": "Maintain technical accuracy while being concise"
}
param_set.update(new_values)

# Parameters in the module are now updated
print(summarization_module._system_prompt)  # New value
```

### Optimization Architecture: LangOptim + LangCritic

Dachi's optimization architecture mirrors traditional ML but operates on text:

```
Traditional ML:          Dachi Text Optimization:
┌─────────────────┐     ┌─────────────────────┐
│ Neural Network  │     │  Module (text params)│
└────────┬────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│  Forward Pass   │     │  forward() execution │
└────────┬────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│   Loss Function │     │  LangCritic + Criterion│
└────────┬────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│  Backprop/SGD   │     │  LangOptim + LLM    │
└────────┬────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│ Update Weights  │     │  Update Text Params  │
└─────────────────┘     └─────────────────────┘
```

**LangCritic**: Evaluates outputs using structured criteria

```python
from dachi.proc import LangCritic
from dachi.inst import PassFailCriterion

criterion = PassFailCriterion(
    name="summarization_quality",
    passed=BoolField(description="Whether summary captures key points"),
    passing_criteria=TextField(description="Explanation of quality assessment")
)

critic = LangCritic(
    criterion=criterion,
    evaluator=my_llm,
    prompt_template="Criterion: {criterion}\nInput: {input}\nOutput: {output}\n\nEvaluate:"
)

# Evaluate outputs
evaluation = critic.forward(
    output=summary,
    input=original_text
)
```

**LangOptim**: Updates text parameters based on evaluations

```python
from dachi.proc import LangOptim

class SummarizationOptimizer(LangOptim):
    def objective(self) -> str:
        return "Optimize text parameters to generate concise, accurate summaries"

    def constraints(self) -> str:
        return "Maintain clarity and preserve key information"

    def param_evaluations(self, evaluations):
        return "\n".join([f"Input: {e.input}\nOutput: {e.output}\nPassed: {e.passed}"
                         for e in evaluations])

optimizer = SummarizationOptimizer(
    llm=my_llm,
    params=param_set,
    criterion=criterion,
    prompt_template="..."
)

# Optimization loop
for iteration in range(10):
    outputs = [module.forward(text) for text in test_texts]
    evaluations = critic.batch_forward(outputs, test_texts)

    if pass_rate(evaluations) >= 0.9:
        break

    optimizer.step(evaluations)  # Updates text parameters via LLM
```

The LangOptim.step() method:
1. Formats evaluations as text feedback
2. Calls LLM with `structure=params.to_schema()` for structured output
3. Receives JSON with new parameter values
4. Updates the ParamSet (and module) with new values

This is **Bayesian updating** - the LLM acts as a learned prior that proposes better parameter values based on evaluation feedback.

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

**ShareableItem Hierarchy and Text Parameters:**

The ShareableItem hierarchy supports text parameter optimization:

```python
from dachi.core import Param, PrivateParam, Runtime, Shared

class OptimizableModule(BaseModule):
    # Regular Param - configuration, not optimizable
    model_name: str = Param(default="gpt-4")

    # PrivateParam - text parameters for optimization
    _system_prompt: Param[str] = PrivateParam(
        default="You are helpful",
        description="System behavior definition"
    )

    # Runtime - computed values, not serialized
    _cache: dict = Runtime(default_factory=dict)

    # Attr - runtime state, serialized
    call_count: int = Attr(default=0)

    # Shared - global state across instances
    global_patterns: dict = Shared(default_factory=dict)
```

**Key Differences:**
- `Param`: Configuration, serialized with spec
- `PrivateParam`: Text parameters, serialized with spec, **optimizable by LangOptim**
- `Runtime`: Temporary state, not serialized
- `Attr`: Instance state, serialized separately
- `Shared`: Global state, serialized separately

The distinction enables:
1. Text parameters can be extracted and optimized without affecting runtime state
2. Optimization updates persist through serialization
3. Runtime caches don't bloat saved state

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
- **Text parameter optimization** through ParamSet + LangOptim + LangCritic
- **Intelligent behavior** through Task coordination
- **Learning** through state persistence
- **Consistent execution** through Process patterns
- **System evolution** through BaseModule serialization

## Architecture Summary

Dachi's architecture centers on **text as parameters**:

```
┌─────────────────────────────────────────────────────┐
│             Text Parameter Optimization              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐    ┌──────────────┐              │
│  │   Module     │───▶│  ParamSet    │              │
│  │ (text params)│    │              │              │
│  └──────────────┘    └──────┬───────┘              │
│                              │                       │
│                              ▼                       │
│  ┌──────────────┐    ┌──────────────┐              │
│  │  LangCritic  │◀───│  LangOptim   │              │
│  │ (evaluation) │    │ (update)     │              │
│  └──────────────┘    └──────────────┘              │
│                                                      │
└─────────────────────────────────────────────────────┘
         ▲                        ▲
         │                        │
┌────────┴────────┐     ┌────────┴────────┐
│  Process        │     │  Behavior Trees │
│  (execution)    │     │  (control flow) │
└─────────────────┘     └─────────────────┘
         ▲                        ▲
         │                        │
         └────────────┬───────────┘
                      │
         ┌────────────┴────────────┐
         │  BaseModule (foundation)│
         │  - Spec/State           │
         │  - Registry             │
         │  - Serialization        │
         └─────────────────────────┘
```

**Key Architectural Patterns:**

1. **Text Parameters**: `PrivateParam` with descriptions enables optimization
2. **ParamSet Extraction**: `module.param_set()` groups parameters for batch updates
3. **Structured Evaluation**: LangCritic + ResponseSpec provides typed feedback
4. **Bayesian Updating**: LangOptim uses LLM to propose better parameter values
5. **Spec/State Separation**: Clean division between configuration and runtime
6. **Process Interface**: Uniform execution (sync/async × regular/streaming)
7. **Behavior Trees**: Composable control flow with TaskStatus
8. **Computational Graphs**: DataFlow DAG composition

See also:
- **[Optimization Guide](optimization-guide.md)** - Complete LangOptim workflow
- **[Behavior Trees](behavior-trees-and-coordination.md)** - Task coordination
- **[Computational Graphs](computational-graphs.md)** - DataFlow composition
- **[Process Framework](process-framework.md)** - Execution patterns
- **[Criterion System](criterion-system.md)** - Structured evaluation