# Computational Graphs (DataFlow)

DataFlow is Dachi's declarative framework for building computational graphs (DAGs). It enables you to compose processes into complex dependency networks where data flows automatically from inputs through transformations to outputs.

## Overview

**DataFlow** is a Directed Acyclic Graph (DAG) that:
- Automatically resolves dependencies between nodes
- Executes independent nodes in parallel using asyncio
- Memoizes results (each node runs once per execution)
- Supports both synchronous and asynchronous processes
- Provides clear, serializable graph structure

## Core Concepts

### Nodes

DataFlow has two types of nodes:

1. **V (Variable)**: Input/root nodes that hold values
2. **Process Nodes**: Computation nodes (Process or AsyncProcess)

### References

**Ref** objects create connections between nodes:

```python
from dachi.proc import DataFlow, Ref

dag = DataFlow()
x_ref = dag.add_inp("x", val=10)  # Returns Ref("x")
doubled = dag.link("double", DoubleProcess(), value=x_ref)  # x_ref connects nodes
```

## Basic Usage

### Simple Linear Pipeline

```python
from dachi.proc import DataFlow, Ref, Process

class MultiplyByTwo(Process):
    def forward(self, x: int) -> int:
        return x * 2

class AddFive(Process):
    def forward(self, x: int) -> int:
        return x + 5

# Build the graph
dag = DataFlow()

# Add input
x_ref = dag.add_inp("x", val=10)

# Link computation nodes
doubled = dag.link("doubled", MultiplyByTwo(), x=x_ref)
result = dag.link("result", AddFive(), x=doubled)

# Set output
dag.set_out("result")

# Execute
output = await dag.aforward()  # ((10 * 2) + 5) = 25
```

### Multiple Inputs and Outputs

```python
class Add(Process):
    def forward(self, a: int, b: int) -> int:
        return a + b

class Multiply(Process):
    def forward(self, x: int, y: int) -> int:
        return x * y

# Build graph with multiple inputs
dag = DataFlow()

a_ref = dag.add_inp("a", val=3)
b_ref = dag.add_inp("b", val=4)

sum_ref = dag.link("sum", Add(), a=a_ref, b=b_ref)
product_ref = dag.link("product", Multiply(), x=a_ref, y=b_ref)

# Multiple outputs - returns tuple
dag.set_out(["sum", "product"])

result = await dag.aforward()  # (7, 12)
```

## Building Graphs

### dag.add_inp(name, val)

Add an input variable to the graph:

```python
dag = DataFlow()
x = dag.add_inp("x", val=5)  # Returns Ref("x")
```

**Parameters:**
- `name`: Unique identifier for this input
- `val`: Default value

**Returns:** `Ref` object referencing this input

### dag.link(name, process, **kwargs)

Add a computation node to the graph:

```python
result = dag.link("compute", MyProcess(), x=input_ref, y=10)
```

**Parameters:**
- `name`: Unique identifier for this node
- `process`: Process or AsyncProcess to execute
- `**kwargs`: Arguments to pass to the process
  - Can be `Ref` objects (dependencies) or literal values

**Returns:** `Ref` object referencing this node

### dag.set_out(outputs)

Specify which nodes to return:

```python
dag.set_out("result")           # Single output - returns value
dag.set_out(["x", "y", "z"])   # Multiple outputs - returns tuple
```

## Advanced Patterns

### Diamond Dependency

Multiple nodes can depend on the same input:

```python
class Square(Process):
    def forward(self, x: int) -> int:
        return x * x

class Cube(Process):
    def forward(self, x: int) -> int:
        return x * x * x

class Add(Process):
    def forward(self, a: int, b: int) -> int:
        return a + b

# Build diamond-shaped graph
dag = DataFlow()

x = dag.add_inp("x", val=3)

# Both operations use the same input
squared = dag.link("squared", Square(), x=x)
cubed = dag.link("cubed", Cube(), x=x)

# Combine results
result = dag.link("sum", Add(), a=squared, b=cubed)

dag.set_out("sum")

output = await dag.aforward()  # (3^2) + (3^3) = 9 + 27 = 36
```

### Branching and Merging

```python
class Split(Process):
    def forward(self, x: int) -> tuple:
        return (x // 2, x // 2)

class ProcessLeft(Process):
    def forward(self, val: int) -> int:
        return val * 2

class ProcessRight(Process):
    def forward(self, val: int) -> int:
        return val + 10

class Merge(Process):
    def forward(self, left: int, right: int) -> int:
        return left + right

# Build branching graph
dag = DataFlow()

x = dag.add_inp("x", val=20)
split_ref = dag.link("split", Split(), x=x)

# Access split results using indexing
left_ref = dag.link("left_val", ProcessLeft(), val=split_ref[0])
right_ref = dag.link("right_val", ProcessRight(), val=split_ref[1])

merged = dag.link("merged", Merge(), left=left_ref, right=right_ref)
dag.set_out("merged")

result = await dag.aforward()  # ((20//2)*2) + ((20//2)+10) = 20 + 20 = 40
```

### Overriding Inputs at Runtime

```python
dag = DataFlow()

x = dag.add_inp("x", val=5)
y = dag.add_inp("y", val=10)

sum_ref = dag.link("sum", Add(), a=x, b=y)
dag.set_out("sum")

# Use default values
result1 = await dag.aforward()  # 5 + 10 = 15

# Override inputs at runtime
x_node = dag.nodes["x"]
y_node = dag.nodes["y"]
result2 = await dag.aforward(by={x_node: 20, y_node: 30})  # 20 + 30 = 50
```

### Temporary Output Override

```python
dag = DataFlow()

x = dag.add_inp("x", val=5)
doubled = dag.link("doubled", MultiplyByTwo(), x=x)
added = dag.link("added", AddFive(), x=doubled)

dag.set_out("added")

# Default output
result1 = await dag.aforward()  # 15

# Override output for this execution only
result2 = await dag.aforward(out_override="doubled")  # 10
result3 = await dag.aforward(out_override=["x", "doubled"])  # (5, 10)
```

## Async and Parallel Execution

DataFlow automatically executes independent nodes in parallel:

```python
import asyncio
from dachi.proc import AsyncProcess

class FetchData(AsyncProcess):
    url: str

    async def aforward(self, query: str) -> dict:
        # Simulate API call
        await asyncio.sleep(0.5)
        return {"url": self.url, "data": f"Results for {query}"}

class MergeResults(Process):
    def forward(self, a: dict, b: dict) -> dict:
        return {"a": a, "b": b}

# Build graph with parallel async operations
dag = DataFlow()

query = dag.add_inp("query", val="search term")

# These two fetches will run in parallel
fetch1 = dag.link("api1", FetchData(url="https://api1.com"), query=query)
fetch2 = dag.link("api2", FetchData(url="https://api2.com"), query=query)

# Merge runs after both fetches complete
merged = dag.link("merged", MergeResults(), a=fetch1, b=fetch2)

dag.set_out("merged")

# Executes fetch1 and fetch2 in parallel (0.5s total, not 1s)
result = await dag.aforward()
```

## Working with Node References

### V and T Nodes

Under the hood, DataFlow uses two node types:

- **V** (Variable): Root nodes with values
- **T** (Task): Process nodes with dependencies

You can also build graphs directly with V and T:

```python
from dachi.proc import V, T, sync_t, async_t

# Direct node creation
x = V(val=5, name="input")
doubled = sync_t(MultiplyByTwo(), _name="doubled", x=x)

# Execute by calling aforward
result = await doubled.aforward()  # 10
```

### Indexing Node Outputs

Access elements of node outputs using indexing:

```python
class ReturnTuple(Process):
    def forward(self) -> tuple:
        return (1, 2, 3)

dag = DataFlow()

tuple_ref = dag.link("tuple", ReturnTuple())

# Access individual elements
first = dag.link("first", IdentityProcess(), value=tuple_ref[0])
second = dag.link("second", IdentityProcess(), value=tuple_ref[1])

dag.set_out(["first", "second"])

result = await dag.aforward()  # (1, 2)
```

## Graph Manipulation

### Sub-Graphs

Create a sub-graph with specific outputs:

```python
dag = DataFlow()

x = dag.add_inp("x", val=10)
a = dag.link("a", ProcessA(), x=x)
b = dag.link("b", ProcessB(), x=a)
c = dag.link("c", ProcessC(), x=b)

# Create sub-graph that only computes up to 'b'
sub_dag = dag.sub(outputs=["b"], by={})
```

### Replacing Nodes

Swap out a process in an existing graph:

```python
dag = DataFlow()

x = dag.add_inp("x", val=5)
result = dag.link("process", OldProcess(), x=x)
dag.set_out("result")

# Replace the process
dag.replace("process", NewProcess())

# Now uses NewProcess instead
output = await dag.aforward()
```

## Practical Examples

### Example 1: Data Processing Pipeline

```python
from dachi.proc import Process, DataFlow, Ref

class LoadCSV(Process):
    filepath: str

    def forward(self) -> list[dict]:
        import csv
        with open(self.filepath) as f:
            return list(csv.DictReader(f))

class FilterRows(Process):
    column: str
    threshold: float

    def forward(self, rows: list[dict]) -> list[dict]:
        return [r for r in rows if float(r[self.column]) >= self.threshold]

class Aggregate(Process):
    column: str

    def forward(self, rows: list[dict]) -> float:
        return sum(float(r[self.column]) for r in rows)

# Build pipeline
dag = DataFlow()

raw_data = dag.link("load", LoadCSV(filepath="sales.csv"))
filtered = dag.link("filter", FilterRows(column="price", threshold=100.0), rows=raw_data)
total = dag.link("total", Aggregate(column="price"), rows=filtered)

dag.set_out("total")

result = await dag.aforward()
print(f"Total sales over $100: ${result}")
```

### Example 2: Multi-Source Aggregation

```python
from dachi.proc import AsyncProcess, Process, DataFlow
import asyncio

class FetchFromAPI(AsyncProcess):
    api_url: str

    async def aforward(self) -> list[int]:
        await asyncio.sleep(0.1)  # Simulate API call
        return [1, 2, 3, 4, 5]

class ComputeSum(Process):
    def forward(self, values: list[int]) -> int:
        return sum(values)

class CombineSums(Process):
    def forward(self, *sums: int) -> int:
        return sum(sums)

# Build graph
dag = DataFlow()

# Fetch from multiple sources in parallel
api1_data = dag.link("api1", FetchFromAPI(api_url="https://api1.com"))
api2_data = dag.link("api2", FetchFromAPI(api_url="https://api2.com"))
api3_data = dag.link("api3", FetchFromAPI(api_url="https://api3.com"))

# Compute sums (also in parallel)
sum1 = dag.link("sum1", ComputeSum(), values=api1_data)
sum2 = dag.link("sum2", ComputeSum(), values=api2_data)
sum3 = dag.link("sum3", ComputeSum(), values=api3_data)

# Combine all sums
total = dag.link("total", CombineSums(), sums=[sum1, sum2, sum3])

dag.set_out("total")

result = await dag.aforward()  # Sum of all values from all APIs
```

### Example 3: Conditional Processing

```python
class CheckThreshold(Process):
    threshold: float

    def forward(self, value: float) -> bool:
        return value > self.threshold

class ProcessHighValue(Process):
    def forward(self, value: float) -> str:
        return f"High value: {value}"

class ProcessLowValue(Process):
    def forward(self, value: float) -> str:
        return f"Low value: {value}"

class ConditionalSelect(Process):
    def forward(self, condition: bool, high: str, low: str) -> str:
        return high if condition else low

# Build conditional graph
dag = DataFlow()

value = dag.add_inp("value", val=75.0)

is_high = dag.link("check", CheckThreshold(threshold=50.0), value=value)
high_msg = dag.link("high", ProcessHighValue(), value=value)
low_msg = dag.link("low", ProcessLowValue(), value=value)

result = dag.link("result", ConditionalSelect(),
                  condition=is_high, high=high_msg, low=low_msg)

dag.set_out("result")

output = await dag.aforward()  # "High value: 75.0"
```

## Integration with Other Dachi Components

### With Behavior Trees

Use DataFlow to build complex data pipelines within behavior tree actions:

```python
from dachi.act.bt import Action, TaskStatus

class DataPipelineAction(Action):
    pipeline: DataFlow

    async def execute(self) -> TaskStatus:
        try:
            result = await self.pipeline.aforward()
            # Store result in context
            self.ctx.scope.set("pipeline_result", result)
            return TaskStatus.SUCCESS
        except Exception:
            return TaskStatus.FAILURE
```

### With Module System

DataFlow is a Module, so it supports:

```python
# Serialization
dag = DataFlow()
# ... build graph ...
spec = dag.to_spec()

# Reconstruction
restored_dag = DataFlow.from_spec(spec)

# State management
state = dag.state_dict()
dag.load_state_dict(state)
```

## Best Practices

1. **Name Nodes Descriptively**: Use clear names that indicate what each node does

2. **Keep Processes Small**: Each process should have a single, clear responsibility

3. **Use Type Hints**: Help with debugging and documentation

4. **Avoid Cycles**: DataFlow is a DAG - circular dependencies will cause errors

5. **Leverage Parallelism**: Structure your graph so independent operations can run in parallel

6. **Test Processes Independently**: Test each process before composing into a graph

7. **Use Ref for Connections**: Always use the Ref returned by `add_inp()` and `link()` to connect nodes

## Common Patterns

### Fan-Out Pattern

One input feeds multiple independent processes:

```python
dag = DataFlow()

input_data = dag.add_inp("data", val=[1, 2, 3, 4, 5])

# Multiple independent transformations
stats = dag.link("stats", ComputeStatistics(), data=input_data)
visualization = dag.link("viz", CreateVisualization(), data=input_data)
export = dag.link("export", ExportToCSV(), data=input_data)

dag.set_out(["stats", "viz", "export"])
```

### Fan-In Pattern

Multiple inputs combine into one process:

```python
dag = DataFlow()

source1 = dag.link("source1", FetchSource1())
source2 = dag.link("source2", FetchSource2())
source3 = dag.link("source3", FetchSource3())

combined = dag.link("combine", MergeSources(),
                    a=source1, b=source2, c=source3)

dag.set_out("combined")
```

### Map-Reduce Pattern

```python
class MapProcess(Process):
    def forward(self, items: list) -> list:
        return [item * 2 for item in items]

class ReduceProcess(Process):
    def forward(self, items: list) -> int:
        return sum(items)

dag = DataFlow()

data = dag.add_inp("data", val=[1, 2, 3, 4, 5])
mapped = dag.link("map", MapProcess(), items=data)
reduced = dag.link("reduce", ReduceProcess(), items=mapped)

dag.set_out("reduced")

result = await dag.aforward()  # 30
```

## Next Steps

- **[Process Framework](process-framework.md)** - Deep dive into creating custom processes
- **[Optimization Guide](optimization-guide.md)** - Use DataFlow with LangOptim for adaptive pipelines
- **[Behavior Trees](behavior-trees-and-coordination.md)** - Combine DataFlow with decision-making logic

---

DataFlow provides a powerful, declarative way to compose processes into complex computational graphs with automatic dependency resolution and parallel execution.
