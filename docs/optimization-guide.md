# Optimization Guide: Text Parameter Optimization

This guide covers Dachi's core innovation: **optimizing text parameters using LLMs** instead of gradient descent on numerical parameters. This enables building adaptive AI systems where prompts, strategies, and instructions evolve based on structured feedback.

## Overview

Traditional machine learning optimizes numerical parameters (weights, biases) using gradient descent. Dachi optimizes **text parameters** (prompts, instructions, strategies) using **Bayesian updating performed by LLMs**.

**Key Components:**
- **Text Parameters**: `Param[str]` fields in modules that can be optimized
- **ParamSet**: Collection of parameters to optimize together
- **LangCritic**: Evaluates outputs using structured criteria (ResponseSpec)
- **LangOptim**: Updates text parameters based on evaluations

## The Core Workflow

```
Module with Text Parameters
         ↓
    Extract ParamSet
         ↓
    Generate Outputs
         ↓
    Evaluate with LangCritic (structured feedback via ResponseSpec)
         ↓
    Optimize with LangOptim (Bayesian update via LLM)
         ↓
    Parameters Updated
```

## Text Parameters

### Defining Text Parameters

Text parameters use `Param[str]` with `PrivateParam`:

```python
from dachi.core import Module, Param, PrivateParam

class PromptModule(Module):
    """Module with optimizable text parameters."""

    _system_prompt: Param[str] = PrivateParam(
        default="You are a helpful assistant",
        description="System prompt that sets behavior"
    )

    _task_instruction: Param[str] = PrivateParam(
        default="Analyze the given text",
        description="Specific instruction for the task"
    )

    def forward(self, input_text: str) -> str:
        # Use parameters in your logic
        prompt = f"{self._system_prompt.data}\n\n{self._task_instruction.data}\n\nInput: {input_text}"
        # ... process with LLM ...
        return response
```

**Key Points:**
- Use underscore prefix: `_system_prompt` (convention for private params)
- Type annotation: `Param[str]`
- Initialize with `PrivateParam(default=..., description=...)`
- Access value via `.data` attribute
- Description helps the optimizer understand purpose

### Extracting ParamSet

Get all optimizable parameters from a module:

```python
module = PromptModule()

# Extract all Param[str] fields
param_set = module.param_set()

# ParamSet contains list of parameters
print(len(param_set))  # 2 (_system_prompt, _task_instruction)

# Access individual parameters
for param in param_set:
    print(f"{param.name}: {param.data}")
```

**What's in a ParamSet?**
- Collection of `Param` objects
- Generates JSON schema for LLM structured output
- Handles updates from optimization
- Maintains parameter metadata (description, name)

## LangCritic: Structured Evaluation

LangCritic evaluates outputs using an LLM and a criterion (ResponseSpec):

### Basic Usage

```python
from dachi.proc import LangCritic
from dachi.inst import PassFailCriterion, BoolField, TextField

# Define what "good" means
criterion = PassFailCriterion(
    name="helpfulness",
    description="Evaluate if response is helpful",
    passed=BoolField(description="Whether response is helpful"),
    passing_criteria=TextField(description="Explanation of evaluation")
)

# Create critic
critic = LangCritic(
    criterion=criterion,
    evaluator=my_llm,  # Your LangModel instance
    prompt_template="""
Criterion: {criterion}

Output to evaluate: {output}

Input that produced it: {input}

Provide your evaluation:
"""
)

# Evaluate an output
evaluation = critic.forward(
    output="Here's how to solve your problem...",
    input="How do I install Python?"
)

print(evaluation.passed)  # True/False
print(evaluation.passing_criteria)  # Explanation
```

### Batch Evaluation

Evaluate multiple outputs at once:

```python
# Multiple outputs from different parameter settings
outputs = [
    "Response with strategy A",
    "Response with strategy B",
    "Response with strategy C"
]

inputs = [
    "Question 1",
    "Question 2",
    "Question 3"
]

# Batch evaluation
batch_result = critic.batch_forward(
    outputs=outputs,
    inputs=inputs
)

# Access individual evaluations
for eval in batch_result.responses:
    print(f"Passed: {eval.passed}, Reason: {eval.passing_criteria}")
```

### Advanced Criteria

Use different criteria for different aspects:

```python
from dachi.inst import (
    LikertCriterion,
    NumericalRatingCriterion,
    ChecklistCriterion,
    BoundInt, BoundFloat, DictField, ListField
)

# Quality rating
quality_criterion = LikertCriterion(
    name="response_quality",
    description="Rate response quality 1-5",
    rating=BoundInt(min_val=1, max_val=5, description="Quality rating")
)

# Detailed scoring
scoring_criterion = NumericalRatingCriterion(
    name="accuracy",
    description="Score accuracy 0-10",
    score=BoundFloat(min_val=0.0, max_val=10.0, description="Accuracy score")
)

# Checklist evaluation
requirements_criterion = ChecklistCriterion(
    name="requirements",
    description="Check if all requirements met",
    items=DictField(value_type=bool, description="Requirement checks"),
    missing_items=ListField(item_type=str, description="Missing requirements")
)

# Use appropriate critic for each aspect
quality_critic = LangCritic(criterion=quality_criterion, evaluator=my_llm, prompt_template="...")
scoring_critic = LangCritic(criterion=scoring_criterion, evaluator=my_llm, prompt_template="...")
requirements_critic = LangCritic(criterion=requirements_criterion, evaluator=my_llm, prompt_template="...")
```

## LangOptim: Parameter Optimization

LangOptim updates text parameters based on evaluation feedback:

### Creating an Optimizer

```python
from dachi.proc import LangOptim
from dachi.inst import PassFailCriterion, BoolField, TextField

# Your module with text parameters
module = PromptModule()

# Extract parameters to optimize
params = module.param_set()

# Define evaluation criterion
criterion = PassFailCriterion(
    name="effectiveness",
    passed=BoolField(description="Whether approach was effective"),
    passing_criteria=TextField(description="Why it worked or didn't")
)

# Create optimizer (must subclass LangOptim and implement abstract methods)
class MyOptimizer(LangOptim):
    def objective(self) -> str:
        return "Improve parameters to maximize effectiveness"

    def constraints(self) -> str:
        return "Keep responses concise and professional"

    def param_evaluations(self, evaluations):
        # Format evaluations for the optimizer prompt
        if isinstance(evaluations, list):
            return "\n".join([
                f"Evaluation {i+1}: Passed={e.passed}, Reason={e.passing_criteria}"
                for i, e in enumerate(evaluations)
            ])
        return f"Passed={evaluations.passed}, Reason={evaluations.passing_criteria}"

optimizer = MyOptimizer(
    llm=my_llm,
    params=params,
    criterion=criterion,
    prompt_template="""
Objective: {objective}

Constraints: {constraints}

Previous Evaluations:
{evaluations}

Based on these evaluations, update the parameters to improve performance.
Provide updated parameters in the required format.
"""
)
```

### Optimization Loop

```python
# Optimization loop
num_iterations = 5
test_inputs = ["query 1", "query 2", "query 3"]

for iteration in range(num_iterations):
    print(f"\n=== Iteration {iteration + 1} ===")

    # Generate outputs with current parameters
    outputs = []
    for test_input in test_inputs:
        output = module.forward(test_input)
        outputs.append(output)

    # Evaluate outputs
    evaluations = critic.batch_forward(
        outputs=outputs,
        inputs=test_inputs
    )

    # Check performance
    pass_rate = sum(1 for e in evaluations.responses if e.passed) / len(evaluations.responses)
    print(f"Pass rate: {pass_rate:.1%}")

    # Stop if performance is good enough
    if pass_rate >= 0.9:
        print("Target performance achieved!")
        break

    # Update parameters using optimizer
    optimizer.step(evaluations)

    # Parameters are automatically updated in module
    print(f"Updated system_prompt: {module._system_prompt.data}")
    print(f"Updated task_instruction: {module._task_instruction.data}")
```

### Async Optimization

For better performance with async LLMs:

```python
async def optimize_async():
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")

        # Generate outputs (async)
        outputs = await asyncio.gather(*[
            module.aforward(test_input)
            for test_input in test_inputs
        ])

        # Evaluate (async)
        evaluations = await critic.batch_aforward(
            outputs=outputs,
            inputs=test_inputs
        )

        # Check performance
        pass_rate = sum(1 for e in evaluations.responses if e.passed) / len(evaluations.responses)
        print(f"Pass rate: {pass_rate:.1%}")

        if pass_rate >= 0.9:
            break

        # Update parameters (async)
        await optimizer.astep(evaluations)

        print(f"Updated parameters: {module._system_prompt.data}")

# Run async optimization
await optimize_async()
```

## Complete Example: Optimizing a Summarization Prompt

```python
from dachi.core import Module, Param, PrivateParam
from dachi.proc import LangCritic, LangOptim
from dachi.inst import PassFailCriterion, BoolField, TextField

# 1. Define module with text parameters
class Summarizer(Module):
    """Summarization module with optimizable prompts."""

    _system_prompt: Param[str] = PrivateParam(
        default="You are a professional summarizer",
        description="System prompt for the summarizer"
    )

    _summary_instruction: Param[str] = PrivateParam(
        default="Summarize the key points in 2-3 sentences",
        description="Instruction for how to summarize"
    )

    llm: 'LangModel'  # Your LLM adapter

    def forward(self, text: str) -> str:
        """Generate summary."""
        prompt = f"{self._system_prompt.data}\n\n{self._summary_instruction.data}\n\nText: {text}"
        summary, _, _ = self.llm.forward(prompt)
        return summary

# 2. Define evaluation criterion
criterion = PassFailCriterion(
    name="summary_quality",
    description="Evaluate summary quality",
    passed=BoolField(description="Whether summary meets quality standards"),
    passing_criteria=TextField(description="Detailed evaluation explanation")
)

# 3. Create critic
critic = LangCritic(
    criterion=criterion,
    evaluator=my_llm,
    prompt_template="""
Criterion: {criterion}

Original Text: {input}

Generated Summary: {output}

Evaluate the summary for:
- Completeness: captures all key points
- Conciseness: brief without unnecessary details
- Accuracy: no misrepresentation

Provide your evaluation:
"""
)

# 4. Create optimizer
class SummarizerOptimizer(LangOptim):
    def objective(self) -> str:
        return "Improve summarization prompts to generate high-quality, concise summaries"

    def constraints(self) -> str:
        return "Summaries must be 2-3 sentences, capture key points, and be accurate"

    def param_evaluations(self, evaluations):
        results = []
        for i, eval in enumerate(evaluations.responses):
            results.append(
                f"Summary {i+1}:\n"
                f"  Passed: {eval.passed}\n"
                f"  Feedback: {eval.passing_criteria}"
            )
        return "\n\n".join(results)

summarizer = Summarizer(llm=my_llm)
optimizer = SummarizerOptimizer(
    llm=my_llm,
    params=summarizer.param_set(),
    criterion=criterion,
    prompt_template="""
Objective: {objective}

Constraints: {constraints}

Previous Results:
{evaluations}

Current Parameters:
{parameters}

Based on the evaluation feedback, suggest improved parameter values that will produce better summaries.
"""
)

# 5. Run optimization
test_texts = [
    "Long article about AI...",
    "Research paper on climate change...",
    "News article about economy..."
]

for iteration in range(5):
    print(f"\n=== Iteration {iteration + 1} ===")

    # Generate summaries
    summaries = [summarizer.forward(text) for text in test_texts]

    # Evaluate
    evaluations = critic.batch_forward(
        outputs=summaries,
        inputs=test_texts
    )

    # Check performance
    pass_rate = sum(1 for e in evaluations.responses if e.passed) / len(evaluations.responses)
    print(f"Pass rate: {pass_rate:.1%}")

    if pass_rate >= 0.9:
        print("Target achieved!")
        break

    # Optimize
    optimizer.step(evaluations)

    # View updated parameters
    print(f"\nUpdated system_prompt:\n{summarizer._system_prompt.data}")
    print(f"\nUpdated summary_instruction:\n{summarizer._summary_instruction.data}")
```

## Best Practices

### 1. Parameter Design

**Good parameter descriptions:**
```python
_prompt: Param[str] = PrivateParam(
    default="Analyze sentiment",
    description="Instruction that determines how sentiment is analyzed"
)
```

**Bad parameter descriptions:**
```python
_prompt: Param[str] = PrivateParam(
    default="Analyze sentiment",
    description="A prompt"  # Too vague for optimizer
)
```

### 2. Criterion Selection

Choose criteria that match your optimization goal:

- **PassFailCriterion**: Binary success/failure (good for MVP)
- **LikertCriterion**: Ordinal quality ratings (1-5 scale)
- **NumericalRatingCriterion**: Continuous scoring (0-10)
- **ChecklistCriterion**: Multiple requirements (feature completeness)
- **AnalyticRubricCriterion**: Multi-dimensional evaluation (complex tasks)

### 3. Evaluation Quality

**Provide context to the critic:**
```python
evaluation = critic.forward(
    output=generated_output,
    input=original_input,  # What generated the output
    reference=expected_output,  # Optional: gold standard
    context={"domain": "medical", "audience": "experts"}  # Additional context
)
```

### 4. Optimization Strategy

**Start simple, iterate:**
```python
# Start with 1-2 parameters
params = ParamSet(params=[module._main_prompt])

# Add more as needed
params = ParamSet(params=[
    module._system_prompt,
    module._task_instruction,
    module._formatting_rules
])
```

**Use diverse test cases:**
```python
# Varied inputs expose different failure modes
test_inputs = [
    "Simple case",
    "Edge case with ambiguity",
    "Complex multi-part input",
    "Input with contradictions"
]
```

### 5. Monitoring Progress

Track metrics across iterations:

```python
history = {
    "pass_rates": [],
    "parameters": [],
    "evaluations": []
}

for iteration in range(num_iterations):
    # ... generate, evaluate ...

    pass_rate = sum(1 for e in evaluations.responses if e.passed) / len(evaluations.responses)
    history["pass_rates"].append(pass_rate)
    history["parameters"].append(module.param_set().to_spec())
    history["evaluations"].append(evaluations)

    # Save checkpoint
    if pass_rate > best_pass_rate:
        best_params = module.param_set().to_spec()
        save_checkpoint(best_params)
```

## Common Patterns

### Pattern 1: Multi-Objective Optimization

Optimize for multiple criteria:

```python
quality_critic = LangCritic(criterion=quality_criterion, ...)
brevity_critic = LangCritic(criterion=brevity_criterion, ...)
accuracy_critic = LangCritic(criterion=accuracy_criterion, ...)

# Evaluate with all critics
quality_evals = quality_critic.batch_forward(outputs, inputs)
brevity_evals = brevity_critic.batch_forward(outputs, inputs)
accuracy_evals = accuracy_critic.batch_forward(outputs, inputs)

# Combine evaluations
combined_feedback = combine_evaluations(quality_evals, brevity_evals, accuracy_evals)

# Optimize based on combined feedback
optimizer.step(combined_feedback)
```

### Pattern 2: A/B Testing Parameters

Compare parameter variations:

```python
# Test two strategies
params_a = module.param_set()
params_a.update({"param_0": {"data": "Strategy A"}})

params_b = module.param_set()
params_b.update({"param_0": {"data": "Strategy B"}})

# Generate outputs with each
outputs_a = [module.forward(inp) for inp in test_inputs]

module.param_set().update(params_b.to_spec())
outputs_b = [module.forward(inp) for inp in test_inputs]

# Evaluate both
evals_a = critic.batch_forward(outputs_a, test_inputs)
evals_b = critic.batch_forward(outputs_b, test_inputs)

# Choose better performing
if pass_rate(evals_a) > pass_rate(evals_b):
    module.param_set().update(params_a.to_spec())
else:
    module.param_set().update(params_b.to_spec())
```

### Pattern 3: Hierarchical Optimization

Optimize high-level then low-level parameters:

```python
# First: optimize system-level parameters
system_optimizer = MyOptimizer(
    params=ParamSet(params=[module._system_prompt]),
    ...
)

for _ in range(3):
    # ... optimize system prompt ...
    system_optimizer.step(evaluations)

# Then: optimize task-specific parameters
task_optimizer = MyOptimizer(
    params=ParamSet(params=[module._task_instruction, module._format_rules]),
    ...
)

for _ in range(5):
    # ... optimize task parameters ...
    task_optimizer.step(evaluations)
```

## Integration with Other Components

### With Behavior Trees

```python
from dachi.act.bt import Action, TaskStatus

class OptimizingAction(Action):
    """Action that improves its parameters over time."""

    module: Module  # Your module with text parameters
    optimizer: LangOptim
    critic: LangCritic

    async def execute(self) -> TaskStatus:
        # Generate output
        output = self.module.forward(self.ctx.get("input"))

        # Evaluate
        evaluation = self.critic.forward(
            output=output,
            input=self.ctx.get("input")
        )

        # Store for optimization
        self.ctx.set("last_evaluation", evaluation)

        # Optimize if needed
        if not evaluation.passed:
            await self.optimizer.astep(evaluation)

        return TaskStatus.SUCCESS if evaluation.passed else TaskStatus.FAILURE
```

### With DataFlow

```python
from dachi.proc import DataFlow, Ref

# Create optimization pipeline
dag = DataFlow()

# Input
input_ref = dag.add_inp("input", val=test_input)

# Generate
module_ref = dag.link("generate", module, input=input_ref)

# Evaluate
eval_ref = dag.link("evaluate", critic, output=module_ref, input=input_ref)

# Optimize
optim_ref = dag.link("optimize", optimizer.step, evaluations=eval_ref)

dag.set_out("optimize")

# Execute pipeline
await dag.aforward()
```

## Next Steps

- **[Criterion System](criterion-system.md)** - Deep dive into evaluation schemas
- **[Process Framework](process-framework.md)** - Build custom processes
- **[LangModel Adapters](langmodel-adapters.md)** - Integrate LLMs

---

Text parameter optimization enables building systems that learn and adapt from structured feedback, going beyond static prompts to truly adaptive AI.
