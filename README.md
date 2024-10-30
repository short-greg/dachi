Here's the updated README skeleton with the name "Dachi":

---

# Dachi - AI Library for Building Intelligent Systems

**Dachi** is designed to provide flexibility and control when building intelligent systems using LLMs (Large Language Models). The library provides developers to interact with large language models and other AI models seamlessly within Python code, to be able to easily process responses from LLMs, and to construct complex behaviors making use of parallelization and other decision processes through the behavior tree paradigm.

## Features

- **Flexible Interaction**: Easily read and interact with LLMs directly in Python code, allowing for more natural and controlled usage.
- **Behavior Trees**: Use behavior trees to build complex behaviors such as decision-making and parallel processes.
- **Customizable Workflows**: Define and customize workflows for LLM interaction, giving you fine-grained control over the AI's behavior.

## Installation

To install **Dachi**, use pip:

```bash
pip install dachi
```

## Getting Started

### Basic Usage

Below is a basic example of how to use Dachi to interact with an LLM:

```python
from dachi import LLM

engine = dachi.act.OpenAIChatModel(model='gpt-4o-mini')

@dachi.signaturefunc(engine=engine)
def summarize(document: str, num_sentences: int) -> str:
    """
    Summarize the document in {num_sentences} sentences.

    # Document

    {document}
    """
    pass

# Instantiate the LLM and read a response
summary = summarize(document, 5)
print(summary)


# instructfunc is an alternative 
@dachi.instructfunc(engine=engine)
def summarize(document: str, num_sentences: int) -> str:
    cue = dachi.Cue('Summarize the document in {num_sentences} sentences')
    cue2 = dachi.Cue("""
    
    # Document

    {document}    
    """)
    dachi.op.join([cue, cue2], '\n')

    return dachi.op.fill(document=document, num_sentences=num_sentences)
    
```

### Advanced usage

You can leverage behavior trees to define more complex interactions and processes with the LLM:

```python
import dachi


# This is the standard way to create a behavior tree

engine = dachi.adapt.openai.OpenAIChatModel(model='gpt-4o-mini')

# Define behavior tree nodes
@dachi.signaturefunc(engine=engine)
def task1(document: str):
    """
    Extract keywords from the doucment passed in.

    # Document

    {document}
    """
    pass

@dachi.signaturefunc(engine=engine)
def task2(document: str, num_sentences: int):
    """
    Write a summary of the document passed in {num_sentences} sentences

    {document}
    """
    pass


# Create a behavior tree with parallel execution
tree = Sango(
    root=Parallel([
        dachi.act.taskf(task1), 
        dachi.act.taskf(task2)
    ])
)

while status != dachi.act.RUNNING:
    status = tree.tick()


# You can also use function decorators to create
# a behavior tree

class Agent(Task):

    def __init__(self):

        self.context = ContextSpawner()
        self.data = Shared()
        self.task1 = SomeTask()
        self.task2 = SomeTask2(self.data)

    @sequencefunc('context.sequence')
    def sequence(self):

        yield y == 'ready'
        yield self.task1
        yield self.task2

    def tick(self) -> TaskStatus:

        return self.sequence.task()()
        
    def reset(self):
        self.context = ContextSpawner()

```

## Roadmap

- **Improve planning**: Add support for proactive planning and better integration of planning systems with LLMs.
- **Add more reading features**: Increase the 
- **Add adapters**: Add adapters for a wider variety.
- **Add evaluation and learning capabilities**: Add the ability for the systems to evaluate the output and learn.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Let me know if you need any additional changes!
