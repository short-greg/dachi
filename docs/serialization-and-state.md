# Serialization and State: Building Systems That Learn

## Why This Matters

Most AI systems forget everything after each run. They're like having amnesia - no learning, no improvement, no accumulated wisdom. Dachi's serialization system changes this completely.

**The Vision:** AI systems that remember what worked, learn from mistakes, and even generate new versions of themselves. Imagine a behavior tree that evolves based on success patterns, or an LLM that outputs specs to create entirely new system architectures.

This isn't just about saving data - it's about creating **intelligent systems that improve over time**.

## How It Works: Spec and State

Every BaseModule separates into two parts:

**Spec**: The blueprint - what the module *is* (its configuration and structure)  
**State**: The memory - what the module has *learned* (runtime data that changes)

```python
from dachi.core import BaseModule, Param, Attr, Shared

class LearningAgent(BaseModule):
    # Spec data - defines what this agent is
    model_name: str = Param(default="gpt-4") 
    temperature: float = Param(default=0.7)
    
    # State data - what this agent learns over time
    conversation_count: int = Attr(default=0)
    successful_strategies: dict = Attr(default_factory=dict)
    
    # Global data - shared across ALL instances
    collective_knowledge: dict = Shared(default_factory=dict)
```

## Basic Usage

### Using an Agent and Watching It Learn

```python
# Create an agent
agent = LearningAgent(model_name="gpt-4", temperature=0.8)

# Use it - state changes as it learns
agent.conversation_count += 1
agent.successful_strategies["greeting"] = "friendly_approach"

print(f"Conversations: {agent.conversation_count}")  # 1
print(f"Strategies: {agent.successful_strategies}")  # {'greeting': 'friendly_approach'}
```

### Saving the Agent's Knowledge

```python
# Get the blueprint (spec)
spec = agent.spec()
print(type(spec))  # <class 'LearningAgentSpec'> - a Pydantic BaseModel
print(spec.model_name)  # "gpt-4"
print(spec.temperature)  # 0.8

# Get what it learned (state)
state = agent.state_dict()
print(state)  
# {
#   'model_name': 'gpt-4',           # Param values
#   'temperature': 0.8,              # Param values  
#   'conversation_count': 1,         # Attr values
#   'successful_strategies': {'greeting': 'friendly_approach'}
# }
```

### Restoring the Agent with All Its Learning

```python
# Later, recreate the agent with all its knowledge
learned_agent = LearningAgent.from_spec(spec)
learned_agent.load_state_dict(state)

# The agent remembers everything
print(learned_agent.conversation_count)  # 1
print(learned_agent.successful_strategies)  # {'greeting': 'friendly_approach'}
```

## ShareableItem Types

### Param - Configuration Parameters

Parameters define what your module *is*. They're part of both the spec and the state:

```python
class ConfigurableProcessor(BaseModule):
    # Basic configuration
    model: str = Param(default="gpt-4")
    temperature: float = Param(default=0.7)
    
    # Complex configuration
    tools: list = Param(default_factory=list)
    settings: dict = Param(default_factory=lambda: {"mode": "smart"})

processor = ConfigurableProcessor(
    model="gpt-4-turbo", 
    temperature=0.9,
    tools=["search", "calculator"]
)

# Params appear in both spec and state_dict
spec = processor.spec()
print(spec.model)  # "gpt-4-turbo"

state = processor.state_dict() 
print(state['model'])  # "gpt-4-turbo"
print(state['temperature'])  # 0.9
```

### Attr - Learning State

Attributes track what your module *learns* during operation:

```python
class LearningSystem(BaseModule):
    name: str = Param(default="system")
    
    # Learning state - changes during operation
    success_count: int = Attr(default=0)
    failure_patterns: dict = Attr(default_factory=dict)
    user_preferences: dict = Attr(default_factory=dict)
    
    def record_outcome(self, success: bool, context: dict):
        if success:
            self.success_count += 1
            # Learn successful patterns
            pattern_key = context.get("interaction_type", "general")
            if pattern_key not in self.user_preferences:
                self.user_preferences[pattern_key] = []
            self.user_preferences[pattern_key].append(context)
        else:
            # Learn from failures
            failure_type = context.get("error_type", "unknown")
            self.failure_patterns[failure_type] = context

system = LearningSystem(name="smart_assistant")

# System learns from interactions
system.record_outcome(True, {"interaction_type": "code_help", "language": "python"})
system.record_outcome(False, {"error_type": "timeout", "query_length": 1000})

# State captures the learning
state = system.state_dict()
print(state['success_count'])  # 1
print(state['user_preferences'])  # {'code_help': [...]}
print(state['failure_patterns'])  # {'timeout': {...}}
```

### Shared - Global Knowledge

Shared data is available to ALL instances of a module type, but is NOT saved automatically:

```python
class CollaborativeAgent(BaseModule):
    agent_id: str = Param()
    
    # Personal state
    personal_tasks: list = Attr(default_factory=list)
    
    # Global knowledge - shared across ALL instances
    team_knowledge: dict = Shared(default_factory=dict)
    best_practices: dict = Shared(default_factory=dict)
    
    def learn_globally(self, knowledge_key: str, knowledge_value: any):
        # This updates knowledge for ALL instances immediately
        self.team_knowledge[knowledge_key] = knowledge_value
    
    def share_best_practice(self, practice: dict):
        practice_id = f"practice_{len(self.best_practices)}"
        self.best_practices[practice_id] = practice

# Create multiple agents
alice = CollaborativeAgent(agent_id="alice")
bob = CollaborativeAgent(agent_id="bob")

# Alice learns something
alice.learn_globally("python_best_practices", {"use_type_hints": True})

# Bob immediately knows it
print(bob.team_knowledge["python_best_practices"])  # {'use_type_hints': True}

# Shared state is NOT in state_dict
print("team_knowledge" in alice.state_dict())  # False

# You must handle Shared state explicitly if you want to save it
shared_knowledge = alice.team_knowledge.copy()  # Manual save
alice.team_knowledge.update(shared_knowledge)   # Manual restore
```

## Advanced State Management

### Complex Learning Patterns

```python
class AdaptiveStrategy(BaseModule):
    # Configuration
    domain: str = Param(default="general")
    learning_rate: float = Param(default=0.01)
    
    # Complex learning state
    strategy_weights: dict = Attr(default_factory=lambda: {
        "analytical": 0.4,
        "creative": 0.3,
        "collaborative": 0.3
    })
    
    # Detailed performance tracking
    performance_history: list = Attr(default_factory=list)
    success_patterns: dict = Attr(default_factory=dict)
    
    def adapt_strategy(self, outcome_type: str, success: bool, context: dict):
        """Learn from outcomes and adapt strategy weights"""
        
        # Record the outcome
        outcome_record = {
            "outcome_type": outcome_type,
            "success": success,
            "context": context,
            "weights_before": self.strategy_weights.copy(),
            "timestamp": time.time()
        }
        
        # Adjust strategy weights based on success
        if success:
            # Increase weight for successful strategy
            if outcome_type in self.strategy_weights:
                self.strategy_weights[outcome_type] += self.learning_rate
                
            # Track successful patterns
            pattern_key = f"{outcome_type}_{context.get('complexity', 'normal')}"
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = []
            self.success_patterns[pattern_key].append(context)
        else:
            # Decrease weight for failed strategy
            if outcome_type in self.strategy_weights:
                self.strategy_weights[outcome_type] -= self.learning_rate * 0.5
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            self.strategy_weights = {
                k: v/total_weight for k, v in self.strategy_weights.items()
            }
        
        # Record final state
        outcome_record["weights_after"] = self.strategy_weights.copy()
        self.performance_history.append(outcome_record)
        
        # Keep only last 100 records for memory efficiency
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

strategy = AdaptiveStrategy(domain="problem_solving", learning_rate=0.05)

# System learns from multiple outcomes
strategy.adapt_strategy("analytical", True, {"complexity": "high", "domain": "math"})
strategy.adapt_strategy("creative", True, {"complexity": "medium", "domain": "design"})  
strategy.adapt_strategy("analytical", False, {"complexity": "low", "domain": "social"})

print("Evolved strategy weights:", strategy.strategy_weights)
print("Success patterns learned:", list(strategy.success_patterns.keys()))

# All learning is preserved in state
state = strategy.state_dict()
print("State contains:", list(state.keys()))
```

### Hierarchical State Management

```python
class MultiLevelLearner(BaseModule):
    name: str = Param()
    
    # Nested modules with their own state
    text_processor: BaseModule = Param()
    decision_maker: BaseModule = Param()
    
    # This module's own learning
    coordination_patterns: dict = Attr(default_factory=dict)
    module_performance: dict = Attr(default_factory=dict)

# Create with nested modules
learner = MultiLevelLearner(
    name="coordinator",
    text_processor=LearningSystem(name="text_proc"),
    decision_maker=AdaptiveStrategy(domain="decisions")
)

# Each sub-module learns independently
learner.text_processor.success_count += 5
learner.decision_maker.adapt_strategy("analytical", True, {"context": "business"})

# This module learns about coordination
learner.coordination_patterns["text_to_decision"] = {"latency": 0.1, "accuracy": 0.95}

# State dict includes ALL nested state with dot notation
state = learner.state_dict()
print("Top level keys:", [k for k in state.keys() if "." not in k])
print("Text processor keys:", [k for k in state.keys() if k.startswith("text_processor.")])
print("Decision maker keys:", [k for k in state.keys() if k.startswith("decision_maker.")])

# Reconstruct with all nested learning preserved
new_learner = MultiLevelLearner.from_spec(learner.spec())
new_learner.load_state_dict(state)

# All nested learning is restored
print(new_learner.text_processor.success_count)  # 5
print(new_learner.decision_maker.strategy_weights)  # Evolved weights
```

## Serialization Methods

### spec() - Get the Blueprint

```python
class MyAgent(BaseModule):
    model: str = Param(default="gpt-4")
    temperature: float = Param(default=0.7)
    tools: list = Param(default_factory=list)
    
    # Learning state
    conversation_count: int = Attr(default=0)
    learned_facts: dict = Attr(default_factory=dict)

agent = MyAgent(
    model="gpt-4-turbo", 
    temperature=0.9, 
    tools=["search", "calculator"]
)

# Get the spec - a Pydantic BaseModel
spec = agent.spec()
print(type(spec).__name__)  # MyAgentSpec

# Spec contains the blueprint for recreation
print(spec.model)         # "gpt-4-turbo"
print(spec.temperature)   # 0.9
print(spec.tools)         # ["search", "calculator"]

# Convert to dict if needed
spec_dict = spec.model_dump()
print(spec_dict)
# {
#   'kind': 'MyAgent',
#   'id': '12345...',
#   'model': 'gpt-4-turbo',
#   'temperature': 0.9,
#   'tools': ['search', 'calculator']
# }
```

### schema() - Get the Spec Class

```python
# Get the Pydantic model class for the spec
spec_class = MyAgent.schema()
print(spec_class)  # <class MyAgentSpec>

# Create specs directly from the schema
new_spec = spec_class(
    kind="MyAgent",
    model="claude-3",
    temperature=0.5
)

# Use to recreate modules
new_agent = MyAgent.from_spec(new_spec)
```

### state_dict() - Get All Variable State

```python
agent = MyAgent(model="gpt-4", temperature=0.7)

# Change some state
agent.conversation_count = 10
agent.learned_facts = {"python": "great", "ai": "powerful"}

# Get complete state - both Param and Attr values
state = agent.state_dict()
print(state)
# {
#   'model': 'gpt-4',                    # Param value
#   'temperature': 0.7,                  # Param value  
#   'tools': [],                         # Param value
#   'conversation_count': 10,            # Attr value
#   'learned_facts': {'python': 'great', 'ai': 'powerful'}  # Attr value
# }

# Control what gets included
param_only = agent.state_dict(train=True, runtime=False)   # Only Param values
attr_only = agent.state_dict(train=False, runtime=True)    # Only Attr values
flat_state = agent.state_dict(recurse=False)               # No nested modules
```

### load_state_dict() - Restore Variable State

```python
# Create a new agent
new_agent = MyAgent(model="gpt-3.5")  # Different initial config

# Load the learned state
new_agent.load_state_dict(state)

# Agent now has the loaded state
print(new_agent.model)              # "gpt-4" (updated from state)
print(new_agent.conversation_count) # 10
print(new_agent.learned_facts)      # {'python': 'great', 'ai': 'powerful'}

# Partial loading
new_agent.load_state_dict(attr_only, train=False, runtime=True)  # Only Attr values
```

### parameters() - Inspect All Parameters

```python
agent = MyAgent()

# Get all Param objects
for param in agent.parameters():
    print(f"Parameter: {param.data}")

# Get with names
for name, param in agent.named_parameters():
    print(f"{name}: {param.data}")

# Non-recursive (this module only)  
for param in agent.parameters(recurse=False):
    print(f"Local parameter: {param.data}")
```

## LLM-Generated Systems

This is where serialization becomes powerful - LLMs can create new system architectures by outputting specs:

### LLM Creates Behavior Trees

```python
from dachi.core import AdaptModule
from dachi.act import Task, Parallel, Sequence

class DynamicBehaviorTree(BaseModule):
    """Behavior tree that can be generated by LLMs"""
    
    # The root task can be any Task type
    root_task: AdaptModule[Task] = Param()
    
    # Learning from execution
    execution_count: int = Attr(default=0)
    success_rate: float = Attr(default=0.0)
    
    def execute(self):
        status = self.root_task.forward()
        self.execution_count += 1
        
        # Update success rate
        if status.success:
            current_successes = self.success_rate * (self.execution_count - 1) + 1
            self.success_rate = current_successes / self.execution_count
        else:
            current_successes = self.success_rate * (self.execution_count - 1)
            self.success_rate = current_successes / self.execution_count
            
        return status

# LLM generates a spec for a new behavior tree
llm_prompt = """
Create a behavior tree spec for monitoring and responding to user input.
The tree should parallel monitor tasks with sequential response tasks.
Output as JSON.
"""

response = llm.forward(Msg(role="user", text=llm_prompt))

# Parse LLM response as behavior tree spec
llm_spec = {
    "kind": "DynamicBehaviorTree", 
    "id": "bt_001",
    "root_task": {
        "kind": "Parallel",
        "id": "root_parallel",
        "tasks": [
            {
                "kind": "MonitorTask",
                "id": "monitor_1", 
                "sensor_type": "user_input",
                "threshold": 0.8
            },
            {
                "kind": "Sequence",
                "id": "response_sequence",
                "tasks": [
                    {"kind": "AnalyzeTask", "id": "analyze_1"},
                    {"kind": "RespondTask", "id": "respond_1", "response_type": "helpful"}
                ]
            }
        ]
    }
}

# Create behavior tree from LLM spec
behavior_tree = DynamicBehaviorTree.from_spec(llm_spec)

# Execute the LLM-generated behavior
for _ in range(10):
    status = behavior_tree.execute()
    print(f"Execution {behavior_tree.execution_count}: {status}, Success rate: {behavior_tree.success_rate:.2f}")
```

### Self-Evolving Systems

```python
class EvolvingSystem(BaseModule):
    """System that uses LLMs to evolve its own architecture"""
    
    # Current system configuration
    strategy_type: str = Param(default="conservative")
    component_weights: dict = Param(default_factory=dict)
    
    # Evolution tracking
    generation: int = Attr(default=1)
    evolution_history: list = Attr(default_factory=list)
    performance_scores: list = Attr(default_factory=list)
    
    def evolve(self, llm, performance_threshold=0.8):
        """Use LLM to evolve system architecture based on performance"""
        
        current_performance = self._calculate_performance()
        
        if current_performance < performance_threshold:
            # System needs evolution
            evolution_prompt = f"""
            Current system performance: {current_performance:.2f} (threshold: {performance_threshold})
            Current configuration: {self.spec().model_dump()}
            Performance history: {self.performance_scores[-5:]}  # Last 5 scores
            Evolution history: {[e['reason'] for e in self.evolution_history[-3:]]}
            
            This system is underperforming. Generate an improved configuration spec.
            Focus on the components that correlate with better performance.
            Output as JSON spec that can recreate the system.
            """
            
            response = llm.forward(Msg(role="user", text=evolution_prompt))
            
            try:
                # Parse LLM's evolutionary suggestion
                evolved_spec = json.loads(response.msg.text)
                
                # Create evolved system for testing
                evolved_system = EvolvingSystem.from_spec(
                    evolved_spec, 
                    # Preserve learning state
                )
                evolved_system.load_state_dict(self.state_dict())
                
                # Test performance improvement
                test_performance = evolved_system._test_performance()
                
                if test_performance > current_performance:
                    # Evolution successful - record it
                    self.evolution_history.append({
                        "generation": self.generation,
                        "old_spec": self.spec().model_dump(),
                        "new_spec": evolved_spec,
                        "performance_gain": test_performance - current_performance,
                        "reason": "LLM-driven architecture evolution"
                    })
                    
                    # Evolve this system
                    for key, value in evolved_spec.items():
                        if hasattr(self, key) and key not in ["kind", "id"]:
                            setattr(self, key, value)
                    
                    self.generation += 1
                    return True, test_performance
                    
            except Exception as e:
                print(f"Evolution failed: {e}")
        
        return False, current_performance

# System evolution in action
system = EvolvingSystem(
    strategy_type="adaptive", 
    component_weights={"analysis": 0.4, "synthesis": 0.6}
)

# Run evolution cycles
for cycle in range(5):
    # System operates and measures performance
    performance = system.run_tasks()
    system.performance_scores.append(performance)
    
    # Try to evolve if performance is low
    evolved, new_performance = system.evolve(llm, performance_threshold=0.75)
    
    if evolved:
        print(f"Cycle {cycle}: System evolved to generation {system.generation}")
        print(f"Performance improved: {performance:.3f} → {new_performance:.3f}")
        print(f"New config: {system.spec().model_dump()}")
    else:
        print(f"Cycle {cycle}: No evolution needed (performance: {performance:.3f})")

# System has potentially evolved its own architecture multiple times
final_spec = system.spec()
print(f"Final evolved system (generation {system.generation}): {final_spec.model_dump()}")
```

## AdaptModule: Dynamic Module Parameters

AdaptModule makes modules themselves into parameters, enabling incredible flexibility:

### Basic AdaptModule Usage

```python
class FlexibleProcessor(BaseModule):
    name: str = Param(default="processor")
    
    # The actual processor can be swapped dynamically
    processor: AdaptModule[BaseModule] = Param()
    
    # Track usage
    process_count: int = Attr(default=0)
    
    def process(self, data):
        result = self.processor.forward(data)
        self.process_count += 1
        return result

# Create with different processors
simple_proc = FlexibleProcessor(
    name="simple",
    processor=SimpleTextProcessor(model="basic")
)

advanced_proc = FlexibleProcessor(
    name="advanced", 
    processor=AdvancedAIProcessor(model="gpt-4", temperature=0.3)
)

# Both have the same interface but completely different behavior
result1 = simple_proc.process("Hello world")
result2 = advanced_proc.process("Hello world") 

# AdaptModule handles serialization automatically
state = advanced_proc.state_dict()
print("AdaptModule state keys:", [k for k in state.keys() if 'processor' in k])
# ['processor._adapted_param', 'processor._adapted.model', ...]
```

### LLM-Generated Module Composition

```python
class AIGeneratedPipeline(BaseModule):
    """Pipeline where LLM designs the processing steps"""
    
    name: str = Param()
    steps: list[AdaptModule[BaseModule]] = Param(default_factory=list)
    
    # Learning from execution
    execution_history: list = Attr(default_factory=list)
    step_performance: dict = Attr(default_factory=dict)
    
    def execute(self, input_data):
        current_data = input_data
        step_results = []
        
        for i, step in enumerate(self.steps):
            start_time = time.time()
            current_data = step.forward(current_data)
            duration = time.time() - start_time
            
            step_info = {
                "step_index": i,
                "step_type": step.adapted.__class__.__name__,
                "duration": duration,
                "output_size": len(str(current_data))
            }
            step_results.append(step_info)
            
            # Track step performance
            step_key = f"step_{i}_{step_info['step_type']}"
            if step_key not in self.step_performance:
                self.step_performance[step_key] = []
            self.step_performance[step_key].append(duration)
        
        execution_record = {
            "input": str(input_data)[:100],  # Truncated for storage
            "steps": step_results,
            "total_time": sum(s["duration"] for s in step_results),
            "final_output": str(current_data)[:100]
        }
        self.execution_history.append(execution_record)
        
        return current_data

# LLM designs a processing pipeline
pipeline_prompt = """
Design a text processing pipeline with these capabilities:
1. Clean and normalize text
2. Extract key information  
3. Analyze sentiment
4. Generate summary
5. Create structured output

Output as a JSON spec for AIGeneratedPipeline with appropriate processor modules.
"""

response = llm.forward(Msg(role="user", text=pipeline_prompt))
pipeline_spec = json.loads(response.msg.text)

# Example LLM-generated spec:
pipeline_spec = {
    "kind": "AIGeneratedPipeline",
    "id": "llm_pipeline_001",
    "name": "intelligent_text_processor",
    "steps": [
        {
            "kind": "AdaptModule",
            "id": "step_1",
            "_adapted_param": {
                "kind": "TextCleaner",
                "id": "cleaner_1",
                "remove_punctuation": True,
                "lowercase": True,
                "remove_extra_spaces": True
            }
        },
        {
            "kind": "AdaptModule", 
            "id": "step_2",
            "_adapted_param": {
                "kind": "InformationExtractor",
                "id": "extractor_1",
                "extract_entities": True,
                "extract_keywords": True,
                "max_keywords": 10
            }
        },
        {
            "kind": "AdaptModule",
            "id": "step_3", 
            "_adapted_param": {
                "kind": "SentimentAnalyzer",
                "id": "sentiment_1",
                "model": "roberta-base-sentiment",
                "confidence_threshold": 0.8
            }
        },
        {
            "kind": "AdaptModule",
            "id": "step_4",
            "_adapted_param": {
                "kind": "TextSummarizer", 
                "id": "summarizer_1",
                "max_length": 100,
                "preserve_key_points": True
            }
        },
        {
            "kind": "AdaptModule",
            "id": "step_5",
            "_adapted_param": {
                "kind": "StructuredOutputGenerator",
                "id": "output_gen_1",
                "output_format": "json",
                "include_metadata": True
            }
        }
    ]
}

# Create pipeline from LLM design
pipeline = AIGeneratedPipeline.from_spec(pipeline_spec)

# Execute the LLM-designed pipeline
text_input = "I absolutely love this new AI system! It's incredibly helpful and intuitive."
result = pipeline.execute(text_input)

print("Pipeline execution completed:")
print(f"Steps executed: {len(pipeline.steps)}")
print(f"Final result: {result}")

# Pipeline learned from execution
print(f"Execution count: {len(pipeline.execution_history)}")
print(f"Step performance data: {list(pipeline.step_performance.keys())}")

# Save the learned pipeline
pipeline_state = pipeline.state_dict()
print(f"Serialized pipeline has {len(pipeline_state)} state keys")
```

## Real-World Learning Examples

### Persistent Conversation Agent

```python
class PersistentChatAgent(BaseModule):
    """Chat agent that learns and remembers across sessions"""
    
    # Configuration
    model_name: str = Param(default="gpt-4")
    personality: str = Param(default="helpful")
    
    # Session state (saved per conversation)
    conversation_history: list = Attr(default_factory=list)
    user_context: dict = Attr(default_factory=dict)
    session_count: int = Attr(default=0)
    
    # Learning state (accumulates over time)
    effective_responses: dict = Attr(default_factory=dict)
    conversation_patterns: dict = Attr(default_factory=dict)
    user_feedback_scores: list = Attr(default_factory=list)
    
    # Global knowledge (shared but not auto-saved)
    conversation_templates: dict = Shared(default_factory=dict)
    
    def chat(self, user_message: str, session_id: str = None) -> str:
        # Start new session if needed
        if session_id and session_id != getattr(self, '_current_session', None):
            self.session_count += 1
            self._current_session = session_id
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": time.time(),
            "session_id": session_id
        })
        
        # Analyze user patterns
        self._analyze_user_message(user_message)
        
        # Generate response using learned patterns
        response = self._generate_contextual_response(user_message)
        
        # Record assistant response
        self.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": time.time(),
            "session_id": session_id
        })
        
        return response
    
    def learn_from_feedback(self, feedback_score: int, last_exchange_count: int = 1):
        """Learn from user feedback on recent exchanges"""
        
        self.user_feedback_scores.append({
            "score": feedback_score,
            "timestamp": time.time(),
            "conversation_length": len(self.conversation_history)
        })
        
        # Extract patterns from high-scoring interactions
        if feedback_score >= 4:  # Good feedback
            recent_exchanges = self.conversation_history[-last_exchange_count*2:]  # User + assistant
            
            for i in range(0, len(recent_exchanges), 2):
                if i+1 < len(recent_exchanges):
                    user_msg = recent_exchanges[i]["content"]
                    assistant_msg = recent_exchanges[i+1]["content"]
                    
                    # Learn effective response patterns
                    user_pattern = self._extract_message_pattern(user_msg)
                    if user_pattern not in self.effective_responses:
                        self.effective_responses[user_pattern] = []
                    
                    self.effective_responses[user_pattern].append({
                        "response": assistant_msg,
                        "feedback_score": feedback_score,
                        "context": self.user_context.copy()
                    })
    
    def save_session(self, filepath: str):
        """Save current agent state including all learning"""
        checkpoint = {
            "spec": self.spec().model_dump(),
            "state": self.state_dict(),
            "shared_knowledge": self.conversation_templates.copy()  # Manual save
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    @classmethod
    def load_session(cls, filepath: str):
        """Load agent with all previous learning"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore agent
        agent = cls.from_spec(checkpoint["spec"])
        agent.load_state_dict(checkpoint["state"])
        
        # Restore shared knowledge
        agent.conversation_templates.update(checkpoint.get("shared_knowledge", {}))
        
        return agent

# Usage example
agent = PersistentChatAgent(personality="friendly_expert")

# Have conversations
response1 = agent.chat("I'm learning Python and struggling with decorators")
response2 = agent.chat("Can you give me a simple example?")

# Provide feedback
agent.learn_from_feedback(5, last_exchange_count=2)  # Excellent interaction

# More conversations
response3 = agent.chat("Now I'm confused about async/await")
agent.learn_from_feedback(3, last_exchange_count=1)  # Okay interaction

# Save learning
agent.save_session("learned_agent.json")

# Later, load the learned agent
learned_agent = PersistentChatAgent.load_session("learned_agent.json")

# Agent remembers everything
print(f"Loaded agent has {len(learned_agent.conversation_history)} messages")
print(f"Feedback scores: {[f['score'] for f in learned_agent.user_feedback_scores]}")
print(f"Learned patterns: {list(learned_agent.effective_responses.keys())}")

# Continue conversation with accumulated learning
response4 = learned_agent.chat("I have another Python question")
# Uses all previous learning to provide better responses
```

## Key Concepts

**Spec vs State**: Spec defines what a module *is* (configuration), State tracks what it *learns* (runtime changes).

**Automatic Registry**: All BaseModule classes auto-register, enabling `from_spec()` to recreate any module from just its spec.

**Shared State is Manual**: Shared data is global but not auto-saved - you manage it explicitly when needed.

**AdaptModule Magic**: Modules become parameters themselves, enabling LLM-designed architectures and dynamic system composition.

**Complete Reconstruction**: `from_spec()` + `load_state_dict()` gives you the full learned system back.

**LLM-Generated Systems**: LLMs can output specs to create entirely new system architectures - true machine learning at the system level.

This serialization system transforms AI from stateless tools into intelligent, learning entities that improve over time and can even evolve their own architectures. Your systems become genuinely intelligent because they remember, learn, and adapt.