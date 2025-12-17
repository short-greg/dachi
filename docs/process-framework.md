# Process Framework

The Process framework is Dachi's foundation for building composable computational operations. Processes are modules that transform inputs into outputs, supporting four execution modes: synchronous, asynchronous, streaming, and async streaming.

## Overview

A **Process** is a Module that implements one or more execution interfaces:

- **Process**: Synchronous execution (`forward()`)
- **AsyncProcess**: Asynchronous execution (`aforward()`)
- **StreamProcess**: Synchronous streaming (`stream()`)
- **AsyncStreamProcess**: Asynchronous streaming (`astream()`)

Processes compose into computational graphs (DataFlow), behavior trees, and complex pipelines.

## Four Execution Modes

### 1. Process (Synchronous)

The simplest form - synchronous input/output transformation:

```python
from dachi.proc import Process

class AddNumbers(Process):
    """Add two numbers together"""

    def forward(self, a: int, b: int) -> int:
        return a + b

# Usage
adder = AddNumbers()
result = adder(5, 3)  # 8
# or
result = adder.forward(5, 3)  # 8
```

**Key Points:**
- Implement `forward(*args, **kwargs) -> Any`
- Can be called directly: `process(...)` calls `forward(...)`
- Blocks until computation completes
- Best for quick, CPU-bound operations

### 2. AsyncProcess (Asynchronous)

For operations that benefit from async/await (I/O-bound, concurrent tasks):

```python
from dachi.proc import AsyncProcess
import asyncio

class FetchData(AsyncProcess):
    """Fetch data from an API"""
    url: str

    async def aforward(self, query: str) -> dict:
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"query": query, "url": self.url, "data": "..."}

# Usage
fetcher = FetchData(url="https://api.example.com")
result = await fetcher.aforward("search term")
```

**Key Points:**
- Implement `async def aforward(*args, **kwargs) -> Any`
- Must be awaited: `await process.aforward(...)`
- Non-blocking, enables concurrency
- Best for I/O operations, network calls, LLM requests

### 3. StreamProcess (Synchronous Streaming)

For operations that yield results incrementally:

```python
from dachi.proc import StreamProcess
from typing import Iterator

class GenerateNumbers(StreamProcess):
    """Generate numbers from start to end"""

    def stream(self, start: int, end: int) -> Iterator[int]:
        for i in range(start, end):
            yield i

# Usage
generator = GenerateNumbers()
for num in generator.stream(0, 5):
    print(num)  # Prints 0, 1, 2, 3, 4
```

**Key Points:**
- Implement `stream(*args, **kwargs) -> Iterator[Any]`
- Returns a generator/iterator
- Synchronous iteration with `for`
- Best for generating sequences, processing large datasets in chunks

### 4. AsyncStreamProcess (Async Streaming)

For asynchronous operations that yield results incrementally:

```python
from dachi.proc import AsyncStreamProcess
from typing import AsyncIterator
import asyncio

class StreamTokens(AsyncStreamProcess):
    """Stream LLM response tokens"""

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield token

# Usage
streamer = StreamTokens()
async for token in streamer.astream("Hi"):
    print(token, end="")  # Prints "Hello world!"
```

**Key Points:**
- Implement `async def astream(*args, **kwargs) -> AsyncIterator[Any]`
- Returns async generator
- Async iteration with `async for`
- Best for streaming LLM responses, real-time data feeds

## Multi-Interface Processes

A single process can implement multiple interfaces:

```python
from dachi.proc import Process, AsyncProcess

class FlexibleCompute(Process, AsyncProcess):
    """Support both sync and async execution"""

    def forward(self, x: int) -> int:
        return x * 2

    async def aforward(self, x: int) -> int:
        # Async version with same logic
        return x * 2

# Usage
proc = FlexibleCompute()
result1 = proc(5)           # Sync: 10
result2 = await proc.aforward(5)  # Async: 10
```

This is useful when integrating with both sync and async codebases.

## Helper Functions

Dachi provides helper functions that work with both processes and regular callables:

### forward() / aforward()

Execute any callable or process:

```python
from dachi.proc import forward, aforward

# Works with processes
result = forward(AddNumbers(), 3, 5)  # 8

# Works with functions
result = forward(lambda x: x * 2, 5)  # 10

# Async version
result = await aforward(FetchData(url="..."), "query")
result = await aforward(lambda x: x + 1, 5)  # 6
```

### stream() / astream()

Stream results from any callable or process:

```python
from dachi.proc import stream, astream

# Works with StreamProcess
for val in stream(GenerateNumbers(), 0, 3):
    print(val)  # 0, 1, 2

# Works with generators
def gen():
    yield 1
    yield 2

for val in stream(gen):
    print(val)  # 1, 2

# Async version
async for token in astream(StreamTokens(), "prompt"):
    print(token)
```

## Process Composition

Processes compose into larger systems through several patterns:

### Sequential Composition

Chain processes where each depends on the previous:

```python
class LoadData(Process):
    def forward(self, filename: str) -> list:
        # Load data from file
        return [1, 2, 3, 4, 5]

class FilterData(Process):
    threshold: int

    def forward(self, data: list) -> list:
        return [x for x in data if x > self.threshold]

class ComputeSum(Process):
    def forward(self, data: list) -> int:
        return sum(data)

# Manual composition
loader = LoadData()
filterer = FilterData(threshold=2)
summer = ComputeSum()

data = loader("data.csv")
filtered = filterer(data)
result = summer(filtered)  # Sum of values > 2
```

### DataFlow Composition

Use DataFlow for complex dependency graphs (see [computational-graphs.md](computational-graphs.md)):

```python
from dachi.proc import DataFlow, Ref

dag = DataFlow()
dag.link("raw", LoadData(), filename="data.csv")
dag.link("filtered", FilterData(threshold=2), data=Ref("raw"))
dag.link("sum", ComputeSum(), data=Ref("filtered"))
dag.set_out(["sum"])

result = await dag.aforward()  # Executes the graph
```

### Wrapper Processes

Create reusable process wrappers:

```python
from dachi.proc import Func, AsyncFunc

# Wrap a regular function as a Process
def multiply(x: int, factor: int) -> int:
    return x * factor

proc = Func(f=multiply, kwargs={"factor": 3})
result = proc(5)  # 15

# Wrap an async function
async def fetch(url: str):
    # ...
    return data

proc = AsyncFunc(f=fetch)
result = await proc.aforward("https://...")
```

## Practical Examples

### Example 1: Data Processing Pipeline

```python
class LoadCSV(Process):
    """Load CSV file into list of dicts"""

    def forward(self, filepath: str) -> list[dict]:
        import csv
        with open(filepath) as f:
            return list(csv.DictReader(f))

class FilterRows(Process):
    """Filter rows by condition"""
    column: str
    min_value: float

    def forward(self, rows: list[dict]) -> list[dict]:
        return [
            row for row in rows
            if float(row[self.column]) >= self.min_value
        ]

class AggregateColumn(Process):
    """Compute sum of a column"""
    column: str

    def forward(self, rows: list[dict]) -> float:
        return sum(float(row[self.column]) for row in rows)

# Compose pipeline
loader = LoadCSV()
filterer = FilterRows(column="price", min_value=100.0)
aggregator = AggregateColumn(column="price")

# Execute
rows = loader("sales.csv")
filtered_rows = filterer(rows)
total = aggregator(filtered_rows)
print(f"Total sales over $100: ${total}")
```

### Example 2: Async API Integration

```python
class APIClient(AsyncProcess):
    """Generic async API client"""
    base_url: str
    api_key: str

    async def aforward(self, endpoint: str, params: dict = None) -> dict:
        import aiohttp

        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                return await resp.json()

class ProcessAPIData(Process):
    """Transform API response"""

    def forward(self, response: dict) -> list[str]:
        return [item["name"] for item in response.get("items", [])]

# Usage
client = APIClient(base_url="https://api.example.com", api_key="...")
processor = ProcessAPIData()

response = await client.aforward("users", params={"limit": 10})
names = processor(response)
```

### Example 3: Streaming Data Generator

```python
class ReadFileByLine(StreamProcess):
    """Stream file line by line"""

    def stream(self, filepath: str):
        with open(filepath) as f:
            for line in f:
                yield line.strip()

class TransformLine(Process):
    """Transform a single line"""

    def forward(self, line: str) -> dict:
        parts = line.split(",")
        return {"id": parts[0], "value": parts[1]}

# Stream and process
reader = ReadFileByLine()
transformer = TransformLine()

for line in reader.stream("large_file.csv"):
    record = transformer(line)
    # Process each record without loading entire file
    print(record)
```

### Example 4: Async Streaming LLM

```python
from dachi.proc import AsyncStreamProcess, LangModel
from typing import AsyncIterator

class StreamingLLM(AsyncStreamProcess):
    """Stream LLM response tokens"""
    model: LangModel  # Your LLM adapter

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        # Use LangModel's streaming interface
        async for chunk, messages, raw in self.model.astream([{"role": "user", "content": prompt}]):
            if chunk:
                yield chunk

# Usage
llm = StreamingLLM(model=my_llm_adapter)

async for token in llm.astream("Write a poem"):
    print(token, end="", flush=True)
```

## Integration with Other Dachi Components

### With Behavior Trees

Processes can be wrapped in behavior tree Actions:

```python
from dachi.act.bt import Action, TaskStatus

class ProcessAction(Action):
    """Wrap a Process as a behavior tree Action"""
    process: Process

    async def execute(self) -> TaskStatus:
        try:
            result = self.process(...)
            return TaskStatus.SUCCESS if result else TaskStatus.FAILURE
        except Exception:
            return TaskStatus.FAILURE
```

### With Module System

Processes are Modules, so they support:

- Serialization: `process.to_spec()` / `Process.from_spec(spec)`
- State management: `process.state_dict()` / `process.load_state_dict(state)`
- Parameter tracking: `process.parameters()` for text parameter optimization

## Best Practices

1. **Choose the Right Interface**:
   - CPU-bound, fast operations → `Process`
   - I/O-bound operations → `AsyncProcess`
   - Large datasets, progressive results → `StreamProcess`
   - Async + progressive results → `AsyncStreamProcess`

2. **Keep Processes Focused**: Each process should do one thing well

3. **Make Processes Stateless**: Prefer passing data through arguments rather than storing in instance state

4. **Use Type Hints**: Help with documentation and IDE support

5. **Compose, Don't Monolith**: Build complex operations from simple processes

6. **Test Processes Independently**: Each process should be testable in isolation

## Common Patterns

### Retry Pattern

```python
class RetryProcess(AsyncProcess):
    """Retry a process on failure"""
    inner: AsyncProcess
    max_retries: int = 3

    async def aforward(self, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return await self.inner.aforward(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Cache Pattern

```python
class CachedProcess(Process):
    """Cache process results"""
    inner: Process
    _cache: dict = {}

    def forward(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            self._cache[key] = self.inner.forward(*args, **kwargs)
        return self._cache[key]
```

### Batch Pattern

```python
class BatchProcess(AsyncProcess):
    """Process items in batches"""
    inner: AsyncProcess
    batch_size: int = 10

    async def aforward(self, items: list):
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self.inner.aforward(batch)
            results.extend(batch_results)
        return results
```

## Next Steps

- **[Computational Graphs](computational-graphs.md)** - Compose processes into DAGs with DataFlow
- **[Behavior Trees](behavior-trees-and-coordination.md)** - Integrate processes with decision-making
- **[Optimization Guide](optimization-guide.md)** - Use LangOptim with processes that have text parameters

---

The Process framework provides a unified interface for building composable, testable computational operations that integrate seamlessly with Dachi's module system, behavior trees, and computational graphs.
