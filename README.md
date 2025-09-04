# Dachi - AI Framework for Building Intelligent Systems

**Dachi** is a comprehensive AI framework for building intelligent systems using Large Language Models (LLMs). It provides flexible interaction with LLMs, task coordination through behavior trees, and customizable workflows for AI agents with robust communication and state management.

## Key Features

- **🔄 Async AI Processing**: Coordinate synchronous behavior trees with asynchronous AI calls using AsyncDispatcher
- **💬 Agent Communication**: Typed message passing between agents using Bulletin message boards
- **🧠 Shared State Management**: Thread-safe shared state with TTL and reactive callbacks via Blackboard
- **🌳 Behavior Trees**: Complex decision trees and state machines for AI agent coordination
- **🔧 Modular Architecture**: Extensible BaseModule system with spec/state separation
- **🤖 Multiple AI Providers**: Built-in support for OpenAI, with extensible adapter system

## Quick Start

Get up and running in under 5 minutes:

```python
from dachi.comm import Blackboard, AsyncDispatcher
from dachi.proc import OpenAIChat
from dachi.utils import Msg

# Create core components
blackboard = Blackboard()
dispatcher = AsyncDispatcher(max_concurrent_requests=2)
ai_processor = OpenAIChat(model="gpt-4", temperature=0.7)

def chat(user_message: str) -> str:
    # Submit async AI request
    message = Msg(content=user_message)
    request_id = dispatcher.submit_proc(ai_processor, message)
    
    # Wait for response
    import time
    while True:
        status = dispatcher.status(request_id)
        if status.is_complete():
            return dispatcher.result(request_id).content
        time.sleep(0.1)

# Use it
response = chat("Hello! Explain quantum computing simply.")
print(response)
```

👉 **[Complete Quick Start Guide](docs/quick-start.md)** - Build a smart task processor in 10 minutes

## Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/your-org/dachi.git
cd dachi
pip install -e .

# Install dependencies
conda activate dachi  # or your preferred environment
pip install -r requirements.txt
```

## Documentation

### Getting Started
- 📚 **[Quick Start Guide](docs/quick-start.md)** - Get running in 10 minutes
- 🏗️ **[Architecture Overview](docs/core-architecture.md)** - Understanding Dachi's design
- 📖 **[Usage Patterns](docs/usage-patterns.md)** - Canonical patterns for core components

### Tutorials  
- 💬 **[Simple Chat Agent](docs/tutorial-simple-chat-agent.md)** - Build a conversational AI with state management
- 🤝 **[Multi-Agent Communication](docs/tutorial-multi-agent-communication.md)** - Coordinate multiple specialized agents
- 🔍 **[Architecture in Practice](docs/architecture-in-practice.md)** - How components work together

### Core Architecture
- 🏗️ **[Communication & Requests](docs/communication-and-requests.md)** - Bulletin, Blackboard, AsyncDispatcher
- 💾 **[Serialization & State](docs/serialization-and-state.md)** - State management patterns
- 🔌 **[Adapters](docs/adapters.md)** - AI provider integration
- 📨 **[Message System](docs/message-system.md)** - Msg/Resp handling

## Core Components

### Communication Layer
```python
from dachi.comm import Blackboard, Bulletin, AsyncDispatcher

# Shared state management
blackboard = Blackboard()
blackboard.set("key", "value", scope="agent_001")

# Agent message passing  
bulletin = Bulletin[MessageType]()
bulletin.publish(message)

# Async AI coordination
dispatcher = AsyncDispatcher(max_concurrent_requests=5)
request_id = dispatcher.submit_proc(ai_processor, message)
```

### Behavior Trees
```python
from dachi.act import Task, TaskStatus, Sequence, Parallel

class AIAnalysisTask(Task):
    def tick(self) -> TaskStatus:
        # Submit async AI request on first tick
        if not self.request_submitted:
            self.request_id = dispatcher.submit_proc(ai_proc, message)
            self.request_submitted = True
            return TaskStatus.RUNNING
        
        # Check status on subsequent ticks
        status = dispatcher.status(self.request_id)
        return TaskStatus.SUCCESS if status.is_complete() else TaskStatus.RUNNING

# Compose complex behaviors
tree = Sequence("analysis_pipeline")
tree.add_child(AIAnalysisTask("analyze"))
tree.add_child(ProcessResultsTask("process"))
```

### AI Processing
```python
from dachi.proc import OpenAIChat
from dachi.utils import Msg

# Create AI processor
ai_processor = OpenAIChat(model="gpt-4", temperature=0.7)

# Process messages
message = Msg(content="Analyze this data", context={"type": "analysis"})
response = ai_processor.forward(message)  # Sync
# or
response = await ai_processor.aforward(message)  # Async
```

## Architecture Highlights

**🏗️ Foundation Layer**
- `BaseModule`: Universal building block with registry and spec/state management
- `Process`: Four execution modes (sync/async × regular/streaming)  
- `Task`: Behavior tree nodes with composable TaskStatus operations
- `ShareableItem`: Param/Attr/Shared hierarchy for different data lifecycles

**💬 Communication Layer**  
- `Blackboard`: Thread-safe shared state with TTL and reactive callbacks
- `Bulletin`: Type-safe message boards for agent coordination
- `AsyncDispatcher`: Sync/async coordination for AI processing
- `Msg/Resp`: Structured AI message handling

**🤖 AI Integration**
- OpenAI Chat Completions API with streaming support
- OpenAI Responses API for reasoning models  
- Extensible adapter system for other providers
- Tool calling and multimodal support

## Development

### Testing
```bash
# Run all tests
pytest tests tests_adapt

# Run specific component tests  
pytest tests/comm/     # Communication components
pytest tests/act/      # Behavior trees
pytest tests/core/     # Core modules
```

### Environment Setup
```bash
# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate dachi

# Build documentation
cd docs && make html
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests tests_adapt`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
