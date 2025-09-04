# Adapters: Your Universal AI Interface

## What Are Adapters?

Adapters solve one of the biggest problems in AI development: **every AI provider has a different API**. OpenAI uses one format, Anthropic uses another, local models use something else entirely. Your code becomes a mess of provider-specific logic that's impossible to maintain.

Dachi adapters give you **one interface that works everywhere**. Write your AI logic once, then switch between GPT-4, Claude, local models, or any other provider by changing a single line of code.

## Why This Matters

**Without adapters:**
```python
# OpenAI
openai_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
text = openai_response.choices[0].message.content

# Anthropic  
anthropic_response = anthropic.Completion.create(
    model="claude-3",
    prompt="\n\nHuman: Hello\n\nAssistant:",
    max_tokens_to_sample=100
)
text = anthropic_response.completion

# Local model
local_response = local_model.generate("Hello", max_length=100)
text = local_response.text

# Three different APIs, three different response formats
```

**With Dachi adapters:**
```python
from dachi.proc import OpenAIChat, AnthropicChat, LocalChat
from dachi.core import Msg

# Same interface everywhere
msg = Msg(role="user", text="Hello")

# Pick your provider - everything else stays the same
llm = OpenAIChat()      # or AnthropicChat(), LocalChat(), etc.
response = llm.forward(msg)
text = response.msg.text   # Always the same access pattern

# Switch providers by changing one line - zero code changes needed
```

This means you can:
- **Test different models easily** - swap providers to compare performance
- **Build provider-agnostic systems** - your code works with any AI provider
- **Add fallbacks seamlessly** - automatically switch to backup providers
- **Future-proof your code** - new providers plug in without breaking changes

## Basic Usage

```python
from dachi.proc import OpenAIChat
from dachi.core import Msg

# Simple chat
llm = OpenAIChat()
msg = Msg(role="user", text="Hello!")
response = llm.forward(msg)
print(response.msg.text)
```

```python
# Streaming works the same way
for chunk in llm.stream(msg):
    print(chunk.delta.text, end="", flush=True)
```

```python
# Async too
response = await llm.aforward(msg)
async for chunk in llm.astream(msg):
    print(chunk.delta.text, end="")
```

## Practical Examples

### Multi-Model Pipeline
```python
class IntelligentWriter:
    def __init__(self):
        # Different models for different tasks
        self.planner = OpenAIChat(model="gpt-4", temperature=0.2)  # Conservative
        self.writer = OpenAIChat(model="gpt-4", temperature=0.8)   # Creative  
        self.editor = OpenAIResp(model="o1-preview")               # Reasoning
    
    def write_article(self, topic: str) -> str:
        # Plan with the analytical model
        plan = self.planner.forward(Msg(role="user", text=f"Outline: {topic}"))
        
        # Write with the creative model  
        draft = self.writer.forward(Msg(role="user", text=f"Write: {plan.msg.text}"))
        
        # Edit with the reasoning model
        final = self.editor.forward(Msg(role="user", text=f"Improve: {draft.msg.text}"))
        
        return final.msg.text

writer = IntelligentWriter()
article = writer.write_article("The Future of AI")
```

### Provider Fallback
```python
class RobustLLM:
    def __init__(self):
        self.primary = OpenAIChat(model="gpt-4")
        self.fallback = OpenAIChat(model="gpt-3.5-turbo")  
        self.last_resort = LocalChat(model="llama-3")
    
    def forward(self, msg):
        for llm in [self.primary, self.fallback, self.last_resort]:
            try:
                return llm.forward(msg)
            except Exception as e:
                print(f"Model failed: {e}")
                continue
        raise Exception("All models failed")

# Automatically tries backup models if primary fails
robust = RobustLLM()  
response = robust.forward(msg)  # Will work even if GPT-4 is down
```

### Rate Limiting
```python
import asyncio

class RateLimitedAdapter:
    def __init__(self, base_adapter, max_concurrent=5):
        self.base = base_adapter
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def aforward(self, msg, **kwargs):
        async with self.semaphore:  # Only 5 concurrent requests
            return await self.base.aforward(msg, **kwargs)

# Wrap any adapter with rate limiting
base_llm = OpenAIChat()
limited_llm = RateLimitedAdapter(base_llm, max_concurrent=3)

# Now automatically limits concurrent requests
tasks = [limited_llm.aforward(msg) for _ in range(100)]
responses = await asyncio.gather(*tasks)  # Only 3 at a time
```

### Smart Caching
```python
import hashlib
import json

class CachingAdapter:
    def __init__(self, base_adapter):
        self.base = base_adapter
        self.cache = {}
    
    def _cache_key(self, msg, **kwargs):
        content = json.dumps({
            "text": msg.text,
            "role": msg.role, 
            "params": sorted(kwargs.items())
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def forward(self, msg, **kwargs):
        key = self._cache_key(msg, **kwargs)
        
        if key in self.cache:
            print("Cache hit - no API call needed!")
            return self.cache[key]
        
        response = self.base.forward(msg, **kwargs)
        self.cache[key] = response
        return response

# Wrap any adapter with caching
base_llm = OpenAIChat()
cached_llm = CachingAdapter(base_llm)

# Identical requests use cached responses
response1 = cached_llm.forward(msg)  # API call
response2 = cached_llm.forward(msg)  # Cache hit!
```

## Built-in Adapters

### OpenAIChat - Standard Models
```python
# Basic usage
llm = OpenAIChat()  # Uses gpt-4 by default

# With parameters
llm = OpenAIChat(
    model="gpt-4-turbo",
    temperature=0.7,
    max_tokens=1000
)

# Azure OpenAI  
llm = OpenAIChat(
    model="gpt-4",
    url="https://your-azure-endpoint.openai.azure.com/"
)
```

### OpenAIResp - Reasoning Models  
```python
# For o1-style reasoning models
llm = OpenAIResp(model="o1-preview")

response = llm.forward(Msg(role="user", text="Solve this complex problem..."))

print(f"Answer: {response.msg.text}")      # Final answer
print(f"Reasoning: {response.thinking}")   # Model's reasoning process
```

## Creating Custom Adapters

The most interesting part is creating your own adapters for new AI providers:

### Simple Custom Adapter
```python
from dachi.proc import LLM, AIAdapt
from dachi.core import Msg, Resp, RespDelta
import httpx

class CustomAIAdapter(LLM, AIAdapt):
    """Adapter for a custom AI API"""
    
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(base_url="https://api.customai.com")
    
    def to_input(self, inp: Msg | BaseDialog, **kwargs) -> dict:
        """Convert Dachi messages to custom API format"""
        if isinstance(inp, Msg):
            # Single message
            prompt = f"{inp.role}: {inp.text}"
        else:
            # Multiple messages (dialog)
            prompt = "\n".join(f"{msg.role}: {msg.text}" for msg in inp)
        
        return {
            "prompt": prompt,
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7)
        }
    
    def from_output(self, output: dict, inp=None) -> Resp:
        """Convert API response back to Dachi format"""
        # Extract the generated text
        generated_text = output["completion"]
        
        # Create Dachi message
        msg = Msg(role="assistant", text=generated_text)
        
        # Return complete response with metadata
        return Resp(
            msg=msg,
            model=output.get("model_used"),
            usage={
                "prompt_tokens": output.get("prompt_tokens", 0),
                "completion_tokens": output.get("completion_tokens", 0),
                "total_tokens": output.get("total_tokens", 0)
            },
            response_id=output.get("request_id")
        )
    
    def forward(self, inp: Msg | BaseDialog, **kwargs) -> Resp:
        """Make the API call"""
        api_input = self.to_input(inp, **kwargs)
        
        response = self.client.post(
            "/v1/completions",
            json=api_input,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        
        return self.from_output(response.json(), inp)

# Use it exactly like built-in adapters
custom_llm = CustomAIAdapter(api_key="your-key", model="custom-model-v1")
response = custom_llm.forward(Msg(role="user", text="Hello!"))
print(response.msg.text)
```

### Advanced Streaming Adapter
```python
class StreamingCustomAdapter(CustomAIAdapter):
    """Add streaming support to custom adapter"""
    
    def from_streamed(self, output: dict, inp=None, prev_resp: Resp = None) -> Resp:
        """Handle streaming chunks with proper accumulation"""
        # Extract new text from this chunk
        chunk_text = output.get("delta", {}).get("content", "")
        
        # Accumulate with previous text
        if prev_resp and prev_resp.msg:
            total_text = prev_resp.msg.text + chunk_text
        else:
            total_text = chunk_text
        
        # Create complete message (accumulated state)
        msg = Msg(role="assistant", text=total_text)
        
        # Create delta (just this chunk)
        delta = RespDelta(
            text=chunk_text,
            finish_reason=output.get("finish_reason"),
            usage=output.get("usage")
        )
        
        # Create or spawn response
        if prev_resp is None:
            resp = Resp(msg=msg, delta=delta)
        else:
            resp = prev_resp.spawn(msg=msg, data=output)
            resp.delta = delta
        
        return resp
    
    def stream(self, inp: Msg | BaseDialog, **kwargs):
        """Streaming implementation"""
        api_input = self.to_input(inp, **kwargs)
        api_input['stream'] = True
        
        prev_resp = None
        
        with self.client.stream(
            "POST", "/v1/completions",
            json=api_input,
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    
                    resp = self.from_streamed(chunk_data, inp, prev_resp)
                    prev_resp = resp
                    yield resp

# Now supports streaming too!
streaming_llm = StreamingCustomAdapter(api_key="your-key")

for chunk in streaming_llm.stream(msg):
    print(chunk.delta.text, end="", flush=True)
```

### Connection Pool Adapter
```python
class PooledAdapter:
    """Distribute requests across multiple adapter instances"""
    
    def __init__(self, adapter_class, pool_size=10, **adapter_kwargs):
        # Create pool of adapter instances
        self.pool = [
            adapter_class(**adapter_kwargs) 
            for _ in range(pool_size)
        ]
        self.current = 0
    
    def _get_adapter(self):
        """Round-robin selection"""
        adapter = self.pool[self.current]
        self.current = (self.current + 1) % len(self.pool)
        return adapter
    
    def forward(self, msg, **kwargs):
        return self._get_adapter().forward(msg, **kwargs)
    
    def stream(self, msg, **kwargs):
        yield from self._get_adapter().stream(msg, **kwargs)
    
    async def aforward(self, msg, **kwargs):
        return await self._get_adapter().aforward(msg, **kwargs)
    
    async def astream(self, msg, **kwargs):
        async for chunk in self._get_adapter().astream(msg, **kwargs):
            yield chunk

# Create pool of 10 OpenAI adapters for high throughput
pooled_llm = PooledAdapter(
    OpenAIChat, 
    pool_size=10, 
    model="gpt-4",
    temperature=0.7
)

# Requests are distributed across the pool
responses = [pooled_llm.forward(msg) for _ in range(100)]
```

## Key Adapter Concepts

**Universal Interface**: All adapters provide the same methods (`forward`, `aforward`, `stream`, `astream`) regardless of the underlying provider.

**Message Translation**: Adapters convert between Dachi's unified `Msg`/`Resp` format and provider-specific APIs.

**Streaming Accumulation**: In streaming, `resp.msg.text` contains the complete accumulated text, while `resp.delta.text` contains only the current chunk.

**Metadata Preservation**: Important information like token usage, model names, and provider-specific data is preserved in the `Resp` object.

**Composable Design**: Adapters can be wrapped and combined to add features like rate limiting, caching, fallbacks, and connection pooling.

The adapter pattern makes your AI code portable, testable, and maintainable. You write your intelligence logic once, then plug in any AI provider without changing your core application code.