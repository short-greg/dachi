# Design Pillars

Dachi is built on a foundation of core principles that guide its architecture and development. These pillars reflect our vision for creating AI systems that are not just functional, but intelligent, adaptive, and maintainable.

## Primary Pillars

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

**Unified patterns:**
```python
# Same interface works with different providers
openai_llm = OpenAIChat(model="gpt-4")
anthropic_llm = AnthropicChat(model="claude-3")
local_llm = LocalLLM(model="llama-3")

# Same response handling
for llm in [openai_llm, anthropic_llm, local_llm]:
    resp = llm.forward(msg)
    print(resp.msg.text)  # Always works the same way
```

**Consistent streaming:**
```python
# Streaming works identically across providers
for chunk in llm.stream(msg):
    print(chunk.delta.text, end="", flush=True)  # Same pattern everywhere
```

### 🔧 Flexible

Composable architecture means you can mix and match components to build exactly what you need. No rigid frameworks or opinionated structures.

**Module composition:**
```python
# Build exactly what you need
pipeline = Sequential([
    InputProcessor(),
    ReasoningModule(llm=gpt4),
    FactChecker(knowledge_base=kb),
    ResponseGenerator(llm=claude),
    OutputFormatter()
])
```

**Behavior tree flexibility:**
```python
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

### 🔍 Transparent

You can always understand what your system is doing, why it made decisions, and how to debug issues.

**Clear state inspection:**
```python
# Always know what's happening
resp = llm.forward(msg)
print(f"Model: {resp.model}")
print(f"Tokens: {resp.usage}")
print(f"Reasoning: {resp.thinking}")  # For reasoning models
print(f"Processing state: {resp.out_store}")  # Internal state

# Debug behavior trees
status = agent.tick()
print(f"Current task: {agent.current_task}")
print(f"Task status: {status}")
print(f"State variables: {agent.get_state()}")
```

**Debuggable execution:**
```python
# Built-in introspection
with agent.debug_mode():
    result = agent.process(input)
    
# See exactly what happened
for step in agent.execution_trace:
    print(f"{step.timestamp}: {step.component} -> {step.action}")
```

---

These pillars work together to create a framework that enables building intelligent systems that learn and improve over time, while maintaining consistency, flexibility, and transparency throughout the development process.

The result: AI systems that are not just functional, but genuinely intelligent and continuously improving.