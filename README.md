Here's the updated README skeleton with the name "Dachi":

---

# Dachi - AI Library for Building Intelligent Systems

**Dachi** is an AI framework for building  building intelligent systems using LLMs (Large Language Models). It provides developers to interact with large language models and other AI models seamlessly within Python code, to be able to easily process responses from LLMs, and to construct complex behaviors making use of parallelization and other decision processes through the behavior tree paradigm.

## Features

- **Flexible Interaction**: Easily read and interact with LLMs directly in Python code, allowing for more natural and controlled usage.
- **Task Coordination**: Use behavior trees to build complex behaviors such as decision-making with multiple agents. Behavior trees make orchestration and coordination straightforward.
- **Customizable Workflows**: Define and customize workflows for LLMs using behavior trees or dynamically generated graphs.

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

engine = dachi.act.OpenAIChatModel(
    model='gpt-4o-mini'
)

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


keywords = dachi.act.Shared()
summarized = dachi.act.Shared()

# Create a behavior tree with parallel execution
tree = Sango(
    root=Parallel([
        dachi.act.taskf(task1, '<Some document>', out=keywords), 
        dachi.act.taskf(task2, '<Some document>', 4, out=summarized)
    ])
)

while status != dachi.act.RUNNING:
    status = tree.tick()


# You can also use function decorators to create
# a behavior tree

class Agent(Task):

    def __init__(self):

        self.context = Context()
        self.data = Shared()
        self.task1 = SomeTask()
        self.task2 = SomeTask2(self.data)
        self._sequence = None

    @sequencemethod()
    def sequence(self):
        """
        This method yields the tasks or the task statuses
        to the caller. The sequence fails on the first failure.
        """

        yield y == 'ready'
        # This does not actually do anything
        yield dachi.act.TaskStatus.SUCCESS
        yield self.task1
        yield self.task2

    def tick(self) -> TaskStatus:
        
        if self._sequence is None:
            self._sequence = self.sequence()

        return self._sequence()
        
    def reset(self):
        self._sequence = None

```

## Roadmap

- **Improve planning**: Add support for proactive planning and better integration of planning systems with LLMs.
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
