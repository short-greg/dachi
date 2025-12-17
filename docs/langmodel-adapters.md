# LangModel Interface

The `LangModel` abstract base class provides a unified interface for integrating LLMs into Dachi. This enables LangOptim and LangCritic to work with any LLM provider through a consistent API.

## Overview

**Purpose**: LangModel is the adapter interface that connects external LLM APIs to Dachi's optimization and evaluation systems.

**Key Point**: LangModel adapters are **integration tools**, not the main feature of Dachi. The framework's value comes from text parameter optimization, behavior trees, and computational graphs - LangModel simply enables LLM integration.

## The LangModel Interface

```python
from dachi.proc import LangModel
from dachi.core import Inp
import typing as t

class LangModel(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def forward(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Synchronous LLM call."""
        pass

    @abstractmethod
    async def aforward(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Async LLM call."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming synchronous LLM call."""
        pass

    @abstractmethod
    async def astream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming async LLM call."""
        pass
```

## Method Signatures

### Return Type: `Tuple[str, List[Inp], Any]`

All methods return a 3-tuple:

1. **str**: The text response from the LLM
2. **List[Inp]**: Message history (for conversation context)
3. **Any**: Raw response object from the LLM API

### Parameters

- **prompt**: Input to the LLM
  - `Inp` type alias: `BaseModel | Dict[str, Any] | str | Msg`
  - Can be single prompt or list of messages

- **structure**: Optional JSON schema for structured output
  - Used by LangCritic for evaluation schemas
  - Used by LangOptim for parameter updates
  - Can be dict or Pydantic model

- **tools**: Optional tool definitions
  - For function calling / tool use
  - Schema format depends on LLM provider

- **kwargs**: Additional provider-specific parameters
  - temperature, max_tokens, etc.

## The Four Methods

### 1. forward() - Synchronous

Standard blocking call:

```python
response_text, messages, raw = llm.forward(
    prompt="Analyze this text",
    temperature=0.7
)
```

### 2. aforward() - Async

Non-blocking async call:

```python
response_text, messages, raw = await llm.aforward(
    prompt="Analyze this text",
    temperature=0.7
)
```

### 3. stream() - Streaming Sync

Generator yielding response chunks:

```python
for chunk_text, messages, raw_chunk in llm.stream(
    prompt="Write a story"
):
    print(chunk_text, end="", flush=True)
```

### 4. astream() - Streaming Async

Async generator yielding chunks:

```python
async for chunk_text, messages, raw_chunk in llm.astream(
    prompt="Write a story"
):
    print(chunk_text, end="", flush=True)
```

## Implementing a LangModel Adapter

Basic structure:

```python
from dachi.proc import LangModel
from dachi.core import Inp, TextMsg
import typing as t

class MyLLMAdapter(LangModel):
    """Adapter for MyLLM API."""

    api_key: str
    model: str = "my-model-v1"

    def forward(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None = None,
        tools: t.Dict | None = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Synchronous call to MyLLM API."""

        # Convert prompt to API format
        api_messages = self._convert_prompt(prompt)

        # Build request
        request = {
            "model": self.model,
            "messages": api_messages,
            **kwargs
        }

        # Add structured output if provided
        if structure:
            request["response_format"] = self._convert_structure(structure)

        # Call API
        import my_llm_sdk
        response = my_llm_sdk.complete(
            api_key=self.api_key,
            **request
        )

        # Extract response
        response_text = response.choices[0].message.content

        # Build message history
        messages = self._build_messages(api_messages, response_text)

        return (response_text, messages, response)

    async def aforward(self, prompt, structure=None, tools=None, **kwargs):
        """Async version."""
        # Similar to forward() but with async API calls
        import my_llm_sdk
        response = await my_llm_sdk.async_complete(...)
        return (response_text, messages, response)

    def stream(self, prompt, structure=None, tools=None, **kwargs):
        """Streaming version."""
        import my_llm_sdk

        api_messages = self._convert_prompt(prompt)
        accumulated_text = ""

        for chunk in my_llm_sdk.stream_complete(...):
            chunk_text = chunk.delta.content or ""
            accumulated_text += chunk_text

            # Yield chunk
            yield (chunk_text, [], chunk)

        # Final yield with complete message history
        messages = self._build_messages(api_messages, accumulated_text)
        yield ("", messages, None)

    async def astream(self, prompt, structure=None, tools=None, **kwargs):
        """Async streaming version."""
        # Similar to stream() but with async iteration
        import my_llm_sdk

        async for chunk in my_llm_sdk.async_stream_complete(...):
            # ... yield chunks ...
            yield (chunk_text, [], chunk)

    def _convert_prompt(self, prompt):
        """Convert Inp format to API message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return [self._convert_single(p) for p in prompt]
        else:
            return [self._convert_single(prompt)]

    def _convert_single(self, inp: Inp):
        """Convert single Inp to API format."""
        if isinstance(inp, str):
            return {"role": "user", "content": inp}
        elif isinstance(inp, TextMsg):
            return {"role": inp.role, "content": inp.text}
        elif isinstance(inp, dict):
            return inp  # Already in API format
        else:
            # Handle Pydantic models
            return inp.model_dump()

    def _convert_structure(self, structure):
        """Convert structure to API's JSON schema format."""
        if hasattr(structure, "model_json_schema"):
            # Pydantic model
            return structure.model_json_schema()
        else:
            # Already a dict
            return structure

    def _build_messages(self, api_messages, response_text):
        """Build message history as List[Inp]."""
        messages = []

        # Convert API messages to Inp format
        for msg in api_messages:
            messages.append(TextMsg(role=msg["role"], text=msg["content"]))

        # Add assistant response
        messages.append(TextMsg(role="assistant", text=response_text))

        return messages
```

## Message Types (Inp)

The `Inp` type alias supports multiple formats:

```python
# Type alias definition
Inp: TypeAlias = BaseModel | Dict[str, Any] | str | Msg

# Examples of valid Inp values:

# 1. String (simple prompt)
prompt = "Analyze sentiment"

# 2. Dict (message format)
prompt = {"role": "user", "content": "Analyze sentiment"}

# 3. TextMsg (Dachi message)
from dachi.core import TextMsg
prompt = TextMsg(role="user", text="Analyze sentiment")

# 4. Pydantic model (structured input)
class MyInput(BaseModel):
    query: str
    context: str

prompt = MyInput(query="...", context="...")

# 5. List of messages (conversation)
prompts = [
    TextMsg(role="system", text="You are helpful"),
    TextMsg(role="user", text="Hello"),
    TextMsg(role="assistant", text="Hi there!"),
    TextMsg(role="user", text="Analyze sentiment")
]
```

**Why this flexibility?**

Different parts of Dachi use different formats:
- LangOptim: Often uses `TextMsg` for system/user prompts
- LangCritic: May use string prompts
- User code: Can use whatever is most convenient

The adapter's job is to convert any `Inp` format to the API's expected format.

## Integration with Dachi Components

### With LangCritic

LangCritic uses the `structure` parameter:

```python
from dachi.proc import LangCritic
from dachi.inst import PassFailCriterion

criterion = PassFailCriterion(...)
llm = MyLLMAdapter(api_key="...")

critic = LangCritic(
    criterion=criterion,
    evaluator=llm,  # Your adapter
    prompt_template="..."
)

# LangCritic calls: llm.forward(prompt, structure=criterion.response_schema)
evaluation = critic.forward(output="...")
```

### With LangOptim

LangOptim uses the `structure` parameter for parameter updates:

```python
from dachi.proc import LangOptim

optimizer = MyOptimizer(
    llm=llm,  # Your adapter
    params=param_set,
    criterion=criterion,
    prompt_template="..."
)

# LangOptim calls: llm.forward(messages, structure=params.to_schema())
optimizer.step(evaluations)
```

## Practical Considerations

### Structured Output

Many modern LLMs support structured output (JSON mode, function calling). Your adapter should:

```python
def _convert_structure(self, structure):
    """Convert Dachi structure to API format."""

    # Get JSON schema
    if hasattr(structure, "model_json_schema"):
        schema = structure.model_json_schema()
    else:
        schema = structure

    # Convert to your API's format
    # OpenAI: {"type": "json_schema", "json_schema": {...}}
    # Anthropic: Different format
    # Others: May vary

    return api_specific_format
```

### Error Handling

Wrap API calls with appropriate error handling:

```python
def forward(self, prompt, **kwargs):
    try:
        response = my_api.complete(...)
        return (response_text, messages, response)
    except my_api.RateLimitError as e:
        raise RuntimeError(f"Rate limit exceeded: {e}")
    except my_api.APIError as e:
        raise RuntimeError(f"API error: {e}")
```

### Message History

The second element of the return tuple (`List[Inp]`) should include:
- All input messages
- The assistant's response

This enables conversation context in multi-turn interactions.

## Testing Your Adapter

Minimal test:

```python
def test_adapter():
    llm = MyLLMAdapter(api_key="test-key", model="test-model")

    # Test simple string
    text, messages, raw = llm.forward("Hello")
    assert isinstance(text, str)
    assert len(messages) > 0

    # Test with structure
    from pydantic import BaseModel

    class Response(BaseModel):
        sentiment: str

    text, messages, raw = llm.forward(
        "Analyze: Great product!",
        structure=Response
    )

    # Should be valid JSON for the structure
    parsed = Response.model_validate_json(text)
    assert parsed.sentiment in ["positive", "negative", "neutral"]
```

## Common Pitfalls

1. **Not handling empty chunks in streaming**
   ```python
   # BAD
   for chunk_text, _, _ in llm.stream(prompt):
       print(chunk_text)  # May print None

   # GOOD
   for chunk_text, _, _ in llm.stream(prompt):
       if chunk_text:
           print(chunk_text, end="")
   ```

2. **Forgetting to build message history**
   ```python
   # BAD - empty message history
   return (response_text, [], raw)

   # GOOD - include conversation
   messages = build_conversation(prompt, response_text)
   return (response_text, messages, raw)
   ```

3. **Not converting structured output correctly**
   - Ensure your adapter extracts pure JSON text from the LLM response
   - Dachi components expect valid JSON string that can be parsed

## Where Adapters Live

**Production adapters**: Should be in your application code or a separate package

**Experimental adapters**: Located in `/local/adapt/` in the Dachi repo (not production-ready)

**Why not in Dachi core?**
- LLM APIs change frequently
- Different users need different providers
- Keeps Dachi focused on the framework, not API integration

## Example: Minimal Mock Adapter

For testing without real API calls:

```python
from dachi.proc import LangModel
from dachi.core import TextMsg
import json

class MockLLM(LangModel):
    """Mock adapter for testing."""

    response: str = "Mock response"

    def forward(self, prompt, structure=None, tools=None, **kwargs):
        # If structure provided, return JSON matching schema
        if structure:
            if hasattr(structure, "model_json_schema"):
                schema = structure.model_json_schema()
            else:
                schema = structure

            # Create minimal valid JSON for schema
            response_text = self._generate_from_schema(schema)
        else:
            response_text = self.response

        # Build messages
        if isinstance(prompt, list):
            messages = prompt + [TextMsg(role="assistant", text=response_text)]
        else:
            messages = [
                TextMsg(role="user", text=str(prompt)),
                TextMsg(role="assistant", text=response_text)
            ]

        return (response_text, messages, {"mock": True})

    async def aforward(self, prompt, structure=None, tools=None, **kwargs):
        return self.forward(prompt, structure, tools, **kwargs)

    def stream(self, prompt, structure=None, tools=None, **kwargs):
        response, messages, raw = self.forward(prompt, structure, tools, **kwargs)
        yield (response, messages, raw)

    async def astream(self, prompt, structure=None, tools=None, **kwargs):
        response, messages, raw = self.forward(prompt, structure, tools, **kwargs)
        yield (response, messages, raw)

    def _generate_from_schema(self, schema):
        """Generate minimal JSON matching schema."""
        # Very basic - real implementation would be more sophisticated
        result = {}
        for prop, spec in schema.get("properties", {}).items():
            if spec.get("type") == "string":
                result[prop] = "mock_value"
            elif spec.get("type") == "boolean":
                result[prop] = True
            elif spec.get("type") == "integer":
                result[prop] = 1
            # ... handle other types ...
        return json.dumps(result)
```

## Next Steps

- **[Optimization Guide](optimization-guide.md)** - See how LangModel is used by LangOptim
- **[Criterion System](criterion-system.md)** - Structured output with LangCritic
- **[Process Framework](process-framework.md)** - LangModel is a Process

---

LangModel provides the integration point between Dachi's optimization framework and external LLM providers. Implement it once, use it everywhere in your Dachi applications.
