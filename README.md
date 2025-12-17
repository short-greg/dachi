# Dachi - ML Framework for Adaptive AI Systems

**Dachi** is a machine learning framework that uses **text as parameters** instead of numerical parameters. Build adaptive AI systems that can modify their own behavior, optimize strategies through Bayesian updating with LLMs, and compose complex decision-making through behavior trees and computational graphs.

## Why Dachi?

Traditional ML frameworks optimize numerical parameters using gradient descent. Dachi optimizes **text parameters** using LLMs for Bayesian updating. This enables:

- **Human-interpretable parameters**: System prompts, strategies, and configurations as text
- **LLM-driven optimization**: Use language models to improve parameters based on structured evaluations
- **Adaptive architectures**: Systems that modify their own structure at runtime
- **Compositional design**: Build complex systems from reusable, interpretable components

## Key Capabilities

### 1. **Dynamic Behavior Trees & State Machines**
Create complex decision-making systems that adapt at runtime:

```python
from dachi.act.bt import SequenceTask, SelectorTask, Action

# Build behavior trees programmatically
decision_tree = SelectorTask(tasks=[
    SequenceTask(tasks=[CheckCondition(), TakeAction()]),
    FallbackAction()
])
```

### 2. **Computational Graphs (DAGs)**
Compose processes into data flow graphs with automatic dependency resolution:

```python
from dachi.proc import DataFlow, Ref

dag = DataFlow()
dag.link("input", LoadData())
dag.link("processed", ProcessData(), data=Ref("input"))
dag.link("output", SaveResults(), data=Ref("processed"))
result = await dag.aforward()
```

### 3. **Text Parameter Optimization**
Optimize prompts, strategies, and configurations using LLMs:

```python
from dachi.core import Module, Param, PrivateParam
from dachi.proc import LangOptim

class Strategy(Module):
    _approach: Param[str] = PrivateParam(default="Be direct and concise")

# Optimize the strategy text using LangOptim
optimizer = LangOptim(llm=my_llm, params=strategy.param_set())
optimizer.step()  # LLM suggests improvements based on evaluations
```

### 4. **Evaluation-Driven Development**
Define what "good" means using structured evaluation schemas:

```python
from dachi.inst import PassFailCriterion, LikertCriterion

criterion = PassFailCriterion()
# Criterion defines evaluation schema for LLM-based assessment
```

### 5. **Modular Architecture**
Composable modules with spec/state separation for serialization and reproducibility:

```python
from dachi.core import Module, ModuleList

class Pipeline(Module):
    stages: ModuleList = [PreprocessModule(), AnalysisModule()]

# Serialize configuration
spec = pipeline.to_spec()
# Save/load state
state = pipeline.state_dict()
```

## Quick Start

```python
from dachi.act.bt import SequenceTask, Action, TaskStatus
from dachi.act.comm import Scope

# Define a custom action
class GreetUser(Action):
    async def execute(self) -> TaskStatus:
        print("Hello from Dachi!")
        return TaskStatus.SUCCESS

# Build and execute a behavior tree
tree = SequenceTask(tasks=[GreetUser()])
scope = Scope()
status = await tree.tick(scope.ctx())
print(f"Status: {status}")  # Status: TaskStatus.SUCCESS
```

👉 **[Complete Quick Start Guide](docs/quick-start.md)** - Learn behavior trees, DataFlow, and text parameters

## Installation

```bash
# Clone and install from source
git clone https://github.com/your-org/dachi.git
cd dachi
pip install -e .

# Or install with conda
conda env create -f environment.yml
conda activate dachi
```

## Documentation

### Core Concepts
- 📚 **[Quick Start Guide](docs/quick-start.md)** - Get running in 10 minutes
- 🏗️ **[Core Architecture](docs/core-architecture.md)** - Understanding the Module system
- 🌳 **[Behavior Trees & Coordination](docs/behavior-trees-and-coordination.md)** - Build decision trees and state machines
- ⚙️ **[Process Framework](docs/process-framework.md)** - Create custom processes with 4 execution modes
- 📊 **[Computational Graphs](docs/computational-graphs.md)** - Compose DAGs with DataFlow

### Advanced Topics
- 🎯 **[Optimization Guide](docs/optimization-guide.md)** - Optimize text parameters using LangOptim
- ✅ **[Criterion System](docs/criterion-system.md)** - Define evaluation schemas
- 💾 **[Serialization & State](docs/serialization-and-state.md)** - State management patterns
- 🔌 **[LangModel Adapters](docs/langmodel-adapters.md)** - Integrate LLM providers (optional)

### Tutorials
- 🔄 **[Adaptive Behavior Trees](docs/tutorial-adaptive-behavior-trees.md)** - Create self-modifying behavior trees
- 🔗 **[Process Composition](docs/tutorial-process-composition.md)** - Build complex computational workflows
- 📝 **[Prompt Optimization](docs/tutorial-prompt-optimization.md)** - Use LangOptim to improve prompts

### Architecture
- 🗣️ **[Communication & Requests](docs/communication-and-requests.md)** - Bulletin, Blackboard, AsyncDispatcher
- 📖 **[Usage Patterns](docs/usage-patterns.md)** - Canonical patterns for core components

## Architecture Overview

Dachi consists of four main layers:

### Core (`dachi/core/`)
**Foundation for text parameters and modules**
- `Module`: Base class with spec/state pattern for serialization
- `Param[T]` / `Runtime[T]` / `Shared[T]`: Different data lifecycle management
- `ParamSet`: Collection of parameters for optimization
- Registry system for module discovery

### Action (`dachi/act/`)
**Decision-making and coordination**
- **Behavior Trees** (`dachi/act/bt/`): `Task`, `TaskStatus`, `SequenceTask`, `SelectorTask`, `ParallelTask`
- **State Machines** (`dachi/act/chart/`): State chart implementation
- **Communication** (`dachi/act/comm/`): `Bulletin` (message passing), `Blackboard` (shared state), `Scope` (contexts)

### Processing (`dachi/proc/`)
**Computational processes and optimization**
- `Process`: Base class with 4 execution modes (sync/async × regular/streaming)
- `DataFlow`: DAG composition with automatic dependency resolution
- `LangOptim`: Optimize text parameters using LLMs
- `LangCritic`: Evaluate outputs using structured schemas
- `AsyncDispatcher`: Coordinate async operations

### Instruction (`dachi/inst/`)
**Evaluation and criteria**
- `ResponseSpec`: Define structured evaluation schemas
- Built-in criteria: `PassFailCriterion`, `LikertCriterion`, `RubricCriterion`, `NarrativeCriterion`, etc.
- Field types for LLM-constrained outputs: `TextField`, `BoolField`, `BoundInt`, etc.

## What Makes Dachi Different?

| Traditional ML | Dachi |
|---------------|-------|
| Numerical parameters (weights, biases) | Text parameters (prompts, strategies) |
| Gradient descent optimization | Bayesian updating via LLMs |
| Black-box models | Interpretable, human-editable parameters |
| Fixed architectures | Adaptive, self-modifying systems |
| Numerical evaluation metrics | Structured, LLM-based evaluations |

## Key Design Principles

1. **Text-Native Optimization**: Parameters are text, not numbers grafted onto text
2. **Compositional**: Modules, processes, and tasks compose cleanly
3. **Adaptive by Design**: Systems can modify their own structure at runtime
4. **Evaluation-Driven**: Structured criteria guide optimization
5. **Async Throughout**: Full support for async execution across all components

## Example Use Cases

### Prompt Engineering
```python
# Optimize a system prompt using LangOptim
class PromptStrategy(Module):
    _system_prompt: Param[str] = PrivateParam(default="Initial prompt")

optimizer = LangOptim(llm=llm, params=strategy.param_set(), criterion=my_criterion)
for _ in range(10):
    optimizer.step()  # LLM improves the prompt based on evaluations
```

### Adaptive Decision Making
```python
# Behavior tree that modifies itself based on performance
def create_strategy(difficulty: str) -> Task:
    if difficulty == "hard":
        return SequenceTask(tasks=[AnalyzeCarefully(), TakeConservativeAction()])
    else:
        return SequenceTask(tasks=[QuickCheck(), TakeAction()])

# Strategy adapts at runtime
current_tree = create_strategy(difficulty_level)
```

### Computational Pipelines
```python
# Build a DAG for data processing
dag = DataFlow()
dag.link("raw", LoadData())
dag.link("clean", CleanData(), data=Ref("raw"))
dag.link("features", ExtractFeatures(), data=Ref("clean"))
dag.link("model_input", FormatForModel(), features=Ref("features"))
dag.set_out(["model_input"])

processed = await dag.aforward()
```

## Development

### Testing
```bash
# Run all tests
pytest tests

# Run specific component tests
pytest tests/act/bt/      # Behavior tree tests
pytest tests/proc/        # Process tests
pytest tests/core/        # Core module tests
```

### Documentation
```bash
# Build Sphinx docs
cd docs && make html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Dachi in your research, please cite:

```bibtex
@software{dachi2024,
  title={Dachi: A Machine Learning Framework for Text Parameter Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/dachi}
}
```

---

**Built with Dachi**: Create adaptive AI systems that learn and improve through structured evaluation and LLM-driven optimization. 🚀
