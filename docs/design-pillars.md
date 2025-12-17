# Design Pillars

Dachi is built on a foundation of core principles that guide its architecture and development. These pillars reflect our vision for creating AI systems that are not just functional, but intelligent, adaptive, and maintainable.

At its core, Dachi is a **machine learning framework that uses text as parameters** instead of numerical parameters, enabling optimization of prompts and instructions through Bayesian updating rather than gradient descent.

## Primary Pillars

### 🎓 Text as Parameters: A New Optimization Paradigm

The most fundamental innovation in Dachi is treating **text as learnable parameters** that can be optimized through evaluation feedback.

**Traditional ML vs Dachi:**

| Aspect | Traditional ML | Dachi |
|--------|---------------|-------|
| Parameters | Numerical weights | Text prompts/instructions |
| Optimization | Gradient descent | Bayesian updating via LLM |
| Feedback | Loss function | Structured evaluation (LangCritic) |
| Update Mechanism | Backpropagation | LangOptim with LLM reasoning |

**What this enables:**

```python
from dachi.core import Module, Param, PrivateParam
from dachi.proc import LangOptim, LangCritic
from dachi.inst import PassFailCriterion

class SummarizationModule(Module):
    # Text parameters - these will be optimized
    _system_prompt: Param[str] = PrivateParam(
        default="You are a helpful summarization assistant",
        description="System prompt defining summarization behavior"
    )

    _instruction: Param[str] = PrivateParam(
        default="Summarize the text concisely",
        description="Task instruction for summarization"
    )

    def forward(self, text: str) -> str:
        # Use current text parameters
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"{self._instruction}\n\n{text}"}
        ]
        response, _, _ = self.llm.forward(messages)
        return response

# Extract parameters for optimization
param_set = module.param_set()

# Define evaluation criterion
criterion = PassFailCriterion(
    name="quality",
    passed=BoolField(description="Summary captures key points"),
    passing_criteria=TextField(description="Quality explanation")
)

# Create optimizer
optimizer = MySummarizationOptimizer(
    llm=my_llm,
    params=param_set,
    criterion=criterion
)

# Optimization loop - just like training a neural network
for iteration in range(10):
    # Generate outputs with current parameters
    outputs = [module.forward(text) for text in test_texts]

    # Evaluate outputs (like computing loss)
    evaluations = critic.batch_forward(outputs, test_texts)

    if pass_rate(evaluations) >= 0.9:
        break

    # Update text parameters (like gradient descent, but for text)
    optimizer.step(evaluations)
```

**Why this matters:**
- **No gradient computation needed** - LLMs reason about what makes better prompts
- **Interpretable parameters** - text parameters are human-readable
- **Domain-agnostic** - works for any text-based task
- **Transfer learning built-in** - LLMs bring prior knowledge about effective prompts

This is **Bayesian optimization**: the LLM acts as a learned prior that proposes better parameter values based on evaluation feedback, similar to how Bayesian optimization uses Gaussian processes for hyperparameter tuning.

### 🧠 Build Intelligent Systems

Dachi exists to help developers create genuinely intelligent systems - not just API wrappers or simple chatbots, but systems capable of complex reasoning, multi-step decision-making, and autonomous behavior.

**What this means:**
- **Multi-agent coordination**: Systems where multiple AI components work together toward shared goals
- **Complex workflows**: Chain together reasoning, planning, execution, and reflection steps
- **Adaptive behavior**: Systems that can adjust their approach based on context and feedback
- **Decision-making frameworks**: Use behavior trees and state machines for sophisticated control flow

**Example: An intelligent code review system**
```python
# This isn't just "send code to LLM" - it's an intelligent system
class CodeReviewAgent(Task):
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.security_checker = SecurityChecker() 
        self.style_checker = StyleChecker()
        self.context = ReviewContext()
    
    async def review(self, code: str) -> ReviewReport:
        # Multi-step intelligent process
        analysis = await self.analyzer.analyze(code, context=self.context)
        security = await self.security_checker.check(code, analysis)
        style = await self.style_checker.check(code, analysis)
        
        # Synthesize findings with reasoning
        return await self.synthesize_report(analysis, security, style)
```

The system reasons about code, maintains context, and produces thoughtful analysis - not just pattern matching.

### 📚 Build Systems That Can Learn

The most important distinction between Dachi and other AI frameworks is the emphasis on systems that persist knowledge and improve over time. Learning isn't an afterthought - it's built into the core architecture.

**Serialization-Enabled Learning:**
- **Spec/State separation**: Clean separation between configuration and learned state
- **Experience persistence**: Systems remember what worked and what didn't
- **Iterative improvement**: Each interaction can update system knowledge
- **Registry-based recovery**: Systems can reconstruct themselves with accumulated learning

**What this enables:**
```python
# System starts simple
agent = CustomerSupportAgent.from_spec({
    "model": "gpt-4",
    "knowledge_base": "basic_faq.json"
})

# Over time, it learns
for customer_interaction in interactions:
    response = agent.handle(customer_interaction)
    feedback = get_customer_feedback(response)
    
    # System updates its understanding
    agent.learn_from_interaction(customer_interaction, response, feedback)
    
    # Knowledge persists
    agent.save_state()

# Later, reconstruct with all learned knowledge
experienced_agent = CustomerSupportAgent.from_spec(
    agent.to_spec(), 
    state=agent.get_state()  # Contains all accumulated learning
)
```

**Learning Patterns:**
- **Successful interaction patterns** → Behavior reinforcement
- **Failed approaches** → Strategy updates  
- **Domain knowledge accumulation** → Expanded capabilities
- **User preference learning** → Personalized responses

This isn't just caching - it's genuine learning that changes system behavior over time.

## Supporting Pillars

### 🎯 Consistent Interface

The same patterns work across different LLMs, processing types, and execution modes. Learn once, apply everywhere.

**Unified execution patterns:**
```python
from dachi.proc import Process

# Same execution interface across all components
class MyProcessor(Process):
    def forward(self, input: str) -> str:
        """Synchronous execution"""
        return self.process(input)

    async def aforward(self, input: str) -> str:
        """Async execution"""
        return await self.async_process(input)

    def stream(self, input: str) -> Iterator[str]:
        """Streaming execution"""
        for chunk in self.stream_process(input):
            yield chunk

    async def astream(self, input: str) -> AsyncIterator[str]:
        """Async streaming"""
        async for chunk in self.async_stream_process(input):
            yield chunk

# Use any execution mode
processor = MyProcessor()
result = processor.forward("input")           # Sync
result = await processor.aforward("input")    # Async
for chunk in processor.stream("input"): ...   # Streaming
```

**Consistent optimization interface:**
```python
# Same optimization pattern works for any task
# Just change the module and criterion

# Summarization
summarizer = SummarizationModule()
param_set = summarizer.param_set()
optimizer = SummarizationOptimizer(llm, param_set, criterion)
optimizer.step(evaluations)

# Classification
classifier = ClassificationModule()
param_set = classifier.param_set()
optimizer = ClassificationOptimizer(llm, param_set, criterion)
optimizer.step(evaluations)

# Same pattern, different domains
```

### 🔧 Flexible

Composable architecture means you can mix and match components to build exactly what you need. No rigid frameworks or opinionated structures.

**Computational graph composition:**
```python
from dachi.proc import DataFlow, V

# Build computational graphs (DAGs)
graph = DataFlow(
    V("input"),
    V("processed") << InputProcessor(V("input")),
    V("analyzed") << Analyzer(V("processed")),
    V("validated") << Validator(V("analyzed")),
    V("output") << Formatter(V("validated"))
)

# Automatic dependency resolution
result = graph.forward(input="raw data")
print(result["output"])
```

**Behavior tree flexibility:**
```python
from dachi.act import Parallel, Selector, Sequence

# Sophisticated control flow
agent_behavior = Parallel([
    # Monitor environment
    MonitorTask(sensors),

    # Make decisions
    Selector([
        EmergencyResponse(),  # High priority
        PlannedAction(),      # Normal operation
        ExploreOptions()      # Fallback
    ])
])
```

**Optimizable module composition:**
```python
# Each module can have text parameters that get optimized together
class Pipeline(Module):
    _preprocessing_instruction: Param[str] = PrivateParam(
        default="Clean and normalize input",
        description="Preprocessing instruction"
    )

    _analysis_prompt: Param[str] = PrivateParam(
        default="Analyze the processed data",
        description="Analysis system prompt"
    )

# Extract and optimize all parameters at once
param_set = pipeline.param_set()  # Gets both parameters
optimizer.step(evaluations)  # Optimizes both together
```

### 🔍 Transparent

You can always understand what your system is doing, why it made decisions, and how to debug issues.

**Clear state inspection:**
```python
# Inspect text parameters
module = SummarizationModule()
print(module._system_prompt)    # See current parameter value
print(module._instruction)

# Inspect evaluation results
evaluation = critic.forward(output, input)
print(f"Passed: {evaluation.passed}")
print(f"Explanation: {evaluation.passing_criteria}")

# Debug behavior trees
status = agent.tick()
print(f"Task status: {status}")
print(f"State variables: {agent.get_state()}")
```

**Transparent optimization:**
```python
# See how parameters change during optimization
optimizer = MyOptimizer(llm, param_set, criterion)

for iteration in range(10):
    # Generate and evaluate
    outputs = [module.forward(text) for text in test_texts]
    evaluations = critic.batch_forward(outputs, test_texts)

    print(f"Iteration {iteration}:")
    print(f"  Pass rate: {pass_rate(evaluations)}")
    print(f"  Current system_prompt: {module._system_prompt}")

    if pass_rate(evaluations) >= 0.9:
        break

    # Update parameters
    optimizer.step(evaluations)

    print(f"  Updated system_prompt: {module._system_prompt}")
```

**Debuggable parameter updates:**
```python
# ParamSet shows exactly what's being optimized
param_set = module.param_set()
print("Parameters to optimize:")
for param in param_set.params:
    print(f"  {param.name}: {param.value}")
    print(f"    Description: {param.description}")
```

---

These pillars work together to create a framework that enables building intelligent systems that learn and improve over time, while maintaining consistency, flexibility, and transparency throughout the development process.

## How the Pillars Work Together

The pillars reinforce each other:

1. **Text as Parameters** provides the optimization foundation
   - Makes learning interpretable and debuggable (**Transparent**)
   - Works with any domain or task (**Flexible**)
   - Uses consistent optimization patterns (**Consistent Interface**)

2. **Intelligent Systems** benefit from optimization
   - Behavior trees can use optimized parameters
   - Multi-agent systems can optimize their prompts
   - Complex workflows improve through parameter tuning

3. **Learning Systems** naturally integrate with optimization
   - State persistence includes optimized parameters
   - Systems improve through both runtime learning and parameter optimization
   - Serialization captures learned parameter values

**The result**: AI systems that are not just functional, but genuinely intelligent, continuously improving through both runtime learning and systematic parameter optimization.

## Next Steps

- **[Quick Start](quick-start.md)** - Build your first optimizable system
- **[Optimization Guide](optimization-guide.md)** - Master text parameter optimization
- **[Core Architecture](core-architecture.md)** - Understand the technical foundation
- **[Behavior Trees](behavior-trees-and-coordination.md)** - Build intelligent control flow