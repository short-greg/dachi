# Quick Start Guide

Get started with Dachi in under 10 minutes. This guide introduces you to Dachi's core capabilities: building adaptive behavior trees, composing computational graphs, and optimizing text parameters.

## What is Dachi?

Dachi is a machine learning framework that uses **text as parameters** instead of numerical parameters. It enables you to:

- **Build adaptive systems** that modify their own behavior
- **Compose computational graphs** (DAGs) from processes
- **Create behavior trees** for complex decision-making
- **Optimize text parameters** using LLMs via Bayesian updating

## Prerequisites

```bash
# Install from source
git clone https://github.com/your-org/dachi.git
cd dachi
pip install -e .
```

## Example 1: Dynamic Behavior Trees

Behavior trees let you create complex decision-making systems. In Dachi, you can build them programmatically and adapt them at runtime.

```python
from dachi.act.bt import SequenceTask, SelectorTask, Action, TaskStatus
from dachi.act.comm import Scope

# Define custom actions
class CheckInventory(Action):
    """Check if item is in stock"""
    item: str = "widget"

    async def execute(self) -> TaskStatus:
        # Simulate inventory check
        in_stock = True  # Your logic here
        return TaskStatus.SUCCESS if in_stock else TaskStatus.FAILURE

class ProcessOrder(Action):
    """Process the order"""

    async def execute(self) -> TaskStatus:
        print("Processing order...")
        return TaskStatus.SUCCESS

class NotifyOutOfStock(Action):
    """Notify customer item is out of stock"""

    async def execute(self) -> TaskStatus:
        print("Item out of stock - notifying customer")
        return TaskStatus.SUCCESS

# Build a behavior tree
order_fulfillment = SelectorTask(tasks=[
    # Try to fulfill order (sequence)
    SequenceTask(tasks=[
        CheckInventory(),
        ProcessOrder()
    ]),
    # If that fails, notify customer
    NotifyOutOfStock()
])

# Execute the behavior tree
scope = Scope()
ctx = scope.ctx()

status = await order_fulfillment.tick(ctx)
print(f"Order fulfillment status: {status}")
```

**Key Points:**
- `SequenceTask`: Executes children in order, fails if any child fails
- `SelectorTask`: Tries each child until one succeeds
- `TaskStatus`: READY, RUNNING, SUCCESS, FAILURE, WAITING
- Build trees dynamically based on runtime conditions

## Example 2: Computational Graphs (DataFlow)

DataFlow lets you build computational graphs where processes depend on each other's outputs.

```python
from dachi.proc import DataFlow, Ref, Process

# Define simple processes
class AddNumbers(Process):
    def forward(self, a: int, b: int) -> int:
        return a + b

class MultiplyNumbers(Process):
    def forward(self, x: int, factor: int) -> int:
        return x * factor

class Constant(Process):
    def __init__(self, value):
        self.value = value

    def forward(self) -> int:
        return self.value

# Build a computational graph
dag = DataFlow()

# Link nodes
dag.link("a", Constant(value=5))
dag.link("b", Constant(value=3))
dag.link("sum", AddNumbers(), a=Ref("a"), b=Ref("b"))
dag.link("result", MultiplyNumbers(), x=Ref("sum"), factor=2)

# Set outputs
dag.set_out(["result"])

# Execute the graph
output = await dag.aforward()
print(f"Result: {output}")  # (16,)  -> (5+3) * 2 = 16
```

**Key Points:**
- `dag.link(name, process, **refs)`: Add a node to the graph
- `Ref("node_name")`: Reference another node's output
- Automatic dependency resolution and caching
- Async execution with `await dag.aforward()`

## Example 3: Text Parameters (Foundation)

Unlike traditional ML frameworks that optimize numerical parameters, Dachi optimizes text.

```python
from dachi.core import Module, Param, PrivateParam

class PromptModule(Module):
    """A module with text parameters"""
    _system_prompt: Param[str] = PrivateParam(
        default="You are a helpful assistant"
    )
    _temperature: Param[float] = PrivateParam(default=0.7)

# Create instance
prompt_mod = PromptModule()

# Access parameters
params = prompt_mod.parameters()
# or
param_set = prompt_mod.param_set()

# Parameters can be optimized using LangOptim (covered in optimization guide)
```

**Key Points:**
- `Param[T]`: Trainable parameter (can be fixed/unfixed)
- `PrivateParam`: Prefix with `_` for private fields
- `parameters()`: Returns all trainable parameters
- `param_set()`: Returns ParamSet for optimization

## What's Next?

Now that you've seen the basics, explore these topics:

### Core Capabilities
- **[Behavior Trees & Coordination](behavior-trees-and-coordination.md)** - Build complex decision trees and state machines
- **[Process Framework](process-framework.md)** - Create custom processes with 4 execution modes
- **[Computational Graphs](computational-graphs.md)** - Build DAGs with DataFlow

### Advanced Topics
- **[Optimization Guide](optimization-guide.md)** - Optimize text parameters using LangOptim
- **[Criterion System](criterion-system.md)** - Define evaluation schemas for optimization
- **[LangModel Adapters](langmodel-adapters.md)** - Integrate LLM providers (optional)

### Tutorials
- **[Adaptive Behavior Trees](tutorial-adaptive-behavior-trees.md)** - Create behavior trees that modify themselves
- **[Process Composition](tutorial-process-composition.md)** - Build complex computational workflows
- **[Prompt Optimization](tutorial-prompt-optimization.md)** - Use LangOptim to improve prompts

## Common Patterns

### Adaptive Behavior Trees

```python
# Build a behavior tree that adapts based on runtime data
def create_strategy_tree(strategy_type: str) -> Task:
    if strategy_type == "aggressive":
        return SequenceTask(tasks=[
            CheckOpportunity(),
            TakeAction()
        ])
    else:
        return SelectorTask(tasks=[
            WaitForSafeCondition(),
            Fallback Action()
        ])

# Strategy can change at runtime
current_strategy = create_strategy_tree("aggressive")
```

### Process Pipelines

```python
# Chain multiple processes together
dag = DataFlow()
dag.link("input", Constant(value="raw data"))
dag.link("cleaned", CleanData(), data=Ref("input"))
dag.link("processed", ProcessData(), data=Ref("cleaned"))
dag.link("output", FormatOutput(), data=Ref("processed"))
dag.set_out(["output"])

result = await dag.aforward()
```

### Module Composition

```python
from dachi.core import ModuleList

class Pipeline(Module):
    """Compose multiple modules"""
    stages: ModuleList = []

    def __init__(self):
        self.stages = ModuleList([
            PreprocessModule(),
            AnalysisModule(),
            OutputModule()
        ])
```

## Key Takeaways

1. **Dachi uses text as parameters** - This enables LLM-driven optimization instead of gradient descent
2. **Build systems dynamically** - Behavior trees, process graphs, and modules can be created and modified at runtime
3. **Compositional architecture** - Modules, processes, and tasks compose into complex systems
4. **Evaluation-driven** - Use the criterion system to define what "good" means for your task

You're now ready to build adaptive AI systems with Dachi! 🚀
