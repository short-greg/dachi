# Behavior Trees and Coordination

Behavior trees provide a powerful framework for creating complex, hierarchical decision-making systems. In Dachi, behavior trees are fully integrated with the Module system, enabling you to build adaptive AI agents that can modify their own decision-making logic at runtime.

## Overview

Behavior trees consist of:
- **Tasks**: Nodes that perform actions or make decisions
- **TaskStatus**: State of task execution (READY, RUNNING, WAITING, SUCCESS, FAILURE)
- **Composite Tasks**: Nodes that control how child tasks execute (sequences, selectors, parallel)
- **Decorators**: Nodes that modify child task behavior

## TaskStatus

The `TaskStatus` enum represents the state of a task:

```python
from dachi.act.bt import TaskStatus

# Five possible states
TaskStatus.READY      # Task hasn't started
TaskStatus.RUNNING    # Task is executing
TaskStatus.WAITING    # Task is waiting (e.g., for async operation)
TaskStatus.SUCCESS    # Task completed successfully
TaskStatus.FAILURE    # Task failed
```

### Status Properties

```python
status = TaskStatus.SUCCESS

# Check status
status.is_done       # True for SUCCESS or FAILURE
status.in_progress   # True for RUNNING or WAITING
status.ready         # True for READY
status.success       # True for SUCCESS
status.failure       # True for FAILURE
status.running       # True for RUNNING
status.waiting       # True for WAITING
```

### Status Composition

TaskStatus supports logical operations following behavior tree semantics:

```python
# OR operation: succeeds if either succeeds
TaskStatus.SUCCESS | TaskStatus.FAILURE  # → SUCCESS
TaskStatus.FAILURE | TaskStatus.FAILURE  # → FAILURE

# AND operation: fails if either fails
TaskStatus.SUCCESS & TaskStatus.FAILURE  # → FAILURE
TaskStatus.SUCCESS & TaskStatus.SUCCESS  # → SUCCESS
```

## Creating Custom Actions

The simplest behavior tree node is an `Action` - a leaf task that performs a specific operation:

```python
from dachi.act.bt import Action, TaskStatus

class CheckHealth(Action):
    """Check if health is above threshold"""
    threshold: float = 50.0

    async def execute(self) -> TaskStatus:
        current_health = self.get_health()  # Your logic here
        if current_health >= self.threshold:
            return TaskStatus.SUCCESS
        return TaskStatus.FAILURE

class HealSelf(Action):
    """Restore health"""
    heal_amount: float = 25.0

    async def execute(self) -> TaskStatus:
        self.restore_health(self.heal_amount)  # Your logic here
        print(f"Healed for {self.heal_amount}")
        return TaskStatus.SUCCESS
```

## Composite Tasks

Composite tasks control how child tasks execute. Dachi provides three main types:

### SequenceTask

Executes children in order. Fails if any child fails. Succeeds only when all children succeed.

```python
from dachi.act.bt import SequenceTask, Action, TaskStatus
from dachi.act.comm import Scope

# Define actions
class OpenDoor(Action):
    async def execute(self) -> TaskStatus:
        print("Opening door...")
        return TaskStatus.SUCCESS

class WalkThrough(Action):
    async def execute(self) -> TaskStatus:
        print("Walking through...")
        return TaskStatus.SUCCESS

class CloseDoor(Action):
    async def execute(self) -> TaskStatus:
        print("Closing door...")
        return TaskStatus.SUCCESS

# Build sequence
enter_room = SequenceTask(tasks=[
    OpenDoor(),
    WalkThrough(),
    CloseDoor()
])

# Execute
scope = Scope()
ctx = scope.ctx()

# First tick: starts OpenDoor
status = await enter_room.tick(ctx)  # RUNNING

# Second tick: OpenDoor completes, starts WalkThrough
status = await enter_room.tick(ctx)  # RUNNING

# Third tick: WalkThrough completes, starts CloseDoor
status = await enter_room.tick(ctx)  # RUNNING

# Fourth tick: CloseDoor completes, sequence succeeds
status = await enter_room.tick(ctx)  # SUCCESS
```

**Use Cases:**
- Multi-step procedures that must execute in order
- Guard conditions followed by actions
- Workflows where later steps depend on earlier ones

### SelectorTask

Tries children in order until one succeeds. Succeeds if any child succeeds. Fails only when all children fail.

```python
from dachi.act.bt import SelectorTask, Action, TaskStatus

class TryDoor(Action):
    async def execute(self) -> TaskStatus:
        if self.door_is_unlocked():
            return TaskStatus.SUCCESS
        return TaskStatus.FAILURE

class PickLock(Action):
    async def execute(self) -> TaskStatus:
        if self.successfully_pick_lock():
            return TaskStatus.SUCCESS
        return TaskStatus.FAILURE

class BreakDown(Action):
    async def execute(self) -> TaskStatus:
        print("Breaking down the door!")
        return TaskStatus.SUCCESS  # Always succeeds

# Try methods in order
enter_building = SelectorTask(tasks=[
    TryDoor(),      # Try first
    PickLock(),     # If door locked, try picking
    BreakDown()     # If picking fails, break it down
])

# Executes until one succeeds or all fail
status = await enter_building.tick(ctx)
```

**Use Cases:**
- Fallback strategies (try A, if fails try B, if fails try C)
- Priority-based decision making
- Robust error handling

### ParallelTask

Executes all children concurrently. Configurable success/failure policies.

```python
from dachi.act.bt import ParallelTask, Action, TaskStatus

class PatrolArea(Action):
    async def execute(self) -> TaskStatus:
        # Patrol logic
        return TaskStatus.RUNNING

class ScanForEnemies(Action):
    async def execute(self) -> TaskStatus:
        # Scanning logic
        return TaskStatus.RUNNING

class MonitorRadio(Action):
    async def execute(self) -> TaskStatus:
        # Radio monitoring logic
        return TaskStatus.RUNNING

# Run all tasks in parallel
guard_duty = ParallelTask(tasks=[
    PatrolArea(),
    ScanForEnemies(),
    MonitorRadio()
])

# All tasks execute concurrently
status = await guard_duty.tick(ctx)
```

**Use Cases:**
- Concurrent monitoring tasks
- Parallel resource gathering
- Multi-tasking agents

## Decorators

Decorators modify the behavior of child tasks:

### Not

Inverts the child's result (SUCCESS ↔ FAILURE):

```python
from dachi.act.bt import Not, Action, TaskStatus

class EnemyNearby(Action):
    async def execute(self) -> TaskStatus:
        if self.detect_enemy():
            return TaskStatus.SUCCESS
        return TaskStatus.FAILURE

# Inverts result
safe_area = Not(task=EnemyNearby())

# If EnemyNearby returns SUCCESS, Not returns FAILURE
# If EnemyNearby returns FAILURE, Not returns SUCCESS
```

### AsLongAs / Until

Loop decorators:

```python
from dachi.act.bt import AsLongAs, Until, Action, TaskStatus

class HasEnergy(Action):
    async def execute(self) -> TaskStatus:
        return TaskStatus.SUCCESS if self.energy > 0 else TaskStatus.FAILURE

class Work(Action):
    async def execute(self) -> TaskStatus:
        self.do_work()
        self.energy -= 10
        return TaskStatus.SUCCESS

# Loop while condition is true
work_cycle = AsLongAs(
    cond=HasEnergy(),
    task=Work()
)

# Or loop until condition becomes true
rest_cycle = Until(
    cond=HasEnergy(),  # Rests until energy is restored
    task=Rest()
)
```

## Building Dynamic Behavior Trees

One of Dachi's key strengths is building behavior trees programmatically:

```python
def create_combat_strategy(difficulty: str, health: float) -> Task:
    """Create adaptive combat behavior based on conditions"""

    if health < 30:
        # Low health: defensive strategy
        return SequenceTask(tasks=[
            TakeCover(),
            HealSelf(),
            SelectorTask(tasks=[
                FleeIfOutnumbered(),
                DefensiveStance()
            ])
        ])

    elif difficulty == "hard":
        # Hard difficulty: cautious approach
        return SequenceTask(tasks=[
            AnalyzeEnemy(),
            SelectorTask(tasks=[
                UseStealthApproach(),
                TacticalEngagement()
            ])
        ])

    else:
        # Normal: aggressive approach
        return SequenceTask(tasks=[
            ChargeForward(),
            AttackEnemy()
        ])

# Create strategy based on current state
current_tree = create_combat_strategy(
    difficulty=game_difficulty,
    health=player.health
)

# Strategy can change every frame
status = await current_tree.tick(ctx)
```

## Advanced Patterns

### Conditional Execution

```python
from dachi.act.bt import PreemptCond, Action, TaskStatus

class LowAmmo(Action):
    async def execute(self) -> TaskStatus:
        return TaskStatus.SUCCESS if self.ammo < 10 else TaskStatus.FAILURE

class Reload(Action):
    async def execute(self) -> TaskStatus:
        self.ammo = 30
        return TaskStatus.SUCCESS

# Only execute if condition is true
conditional_reload = PreemptCond(
    cond=LowAmmo(),
    task=Reload()
)
```

### State-Based Trees

```python
from dachi.core import Module, Runtime, PrivateRuntime

class AgentBehavior(Module):
    """Behavior tree with state"""
    _current_state: Runtime[str] = PrivateRuntime(default="idle")
    _health: Runtime[float] = PrivateRuntime(default=100.0)

    def get_behavior_tree(self) -> Task:
        """Generate behavior tree based on current state"""

        if self._current_state.data == "combat":
            return self._combat_tree()
        elif self._current_state.data == "exploration":
            return self._exploration_tree()
        else:
            return self._idle_tree()

    def _combat_tree(self) -> Task:
        return SequenceTask(tasks=[
            DetectEnemy(),
            EngageEnemy()
        ])

    def _exploration_tree(self) -> Task:
        return SequenceTask(tasks=[
            PickDestination(),
            Navigate(),
            SearchArea()
        ])

    def _idle_tree(self) -> Task:
        return SelectorTask(tasks=[
            CheckForNewMission(),
            Wander(),
            Rest()
        ])
```

### Reusable Subtrees

```python
# Define reusable components
def create_pathfinding_tree(destination) -> Task:
    """Reusable pathfinding behavior"""
    return SequenceTask(tasks=[
        CalculatePath(destination=destination),
        SelectorTask(tasks=[
            UseExistingPath(),
            GenerateNewPath()
        ]),
        FollowPath()
    ])

def create_interaction_tree(target) -> Task:
    """Reusable interaction behavior"""
    return SequenceTask(tasks=[
        ApproachTarget(target=target),
        Interact(target=target)
    ])

# Compose into larger trees
quest_tree = SequenceTask(tasks=[
    create_pathfinding_tree(destination="village"),
    create_interaction_tree(target="quest_giver"),
    create_pathfinding_tree(destination="dungeon"),
    CompleteDungeon(),
    create_pathfinding_tree(destination="village"),
    create_interaction_tree(target="quest_giver")
])
```

## Context and Scopes

Behavior trees execute within a `Scope` which provides shared context:

```python
from dachi.act.comm import Scope

# Create a scope
scope = Scope()

# Store data in scope
scope.set("player_position", (10, 20))
scope.set("enemies_detected", ["enemy1", "enemy2"])

# Tasks can access scope data
class CheckPosition(Action):
    async def execute(self) -> TaskStatus:
        ctx = self.ctx  # Access current context
        position = ctx.scope.get("player_position")
        print(f"Current position: {position}")
        return TaskStatus.SUCCESS

# Execute with context
ctx = scope.ctx()
status = await tree.tick(ctx)
```

## Best Practices

1. **Keep Actions Small**: Each action should do one thing well
2. **Use Selectors for Fallbacks**: Robust systems try multiple approaches
3. **Use Sequences for Procedures**: Multi-step processes with dependencies
4. **Build Dynamically**: Generate trees based on runtime conditions
5. **Compose Reusable Subtrees**: Create libraries of common behaviors
6. **Test Individual Actions**: Unit test actions before composing into trees

## Integration with Other Dachi Components

### With Processes

```python
from dachi.proc import Process

class AIDecisionProcess(Process):
    def forward(self, observation) -> Task:
        """Generate behavior tree from observation"""
        if observation["threat_level"] > 0.7:
            return create_defensive_tree()
        else:
            return create_exploration_tree()

# Use in DataFlow
dag = DataFlow()
dag.link("observation", GetObservation())
dag.link("behavior", AIDecisionProcess(), observation=Ref("observation"))
```

### With Optimization

```python
from dachi.core import Module, Param, PrivateParam

class OptimizableStrategy(Module):
    """Behavior strategy with optimizable text parameters"""
    _approach: Param[str] = PrivateParam(
        default="Be aggressive and direct"
    )
    _fallback: Param[str] = PrivateParam(
        default="Retreat if outnumbered"
    )

    def build_tree(self) -> Task:
        # Use optimized text parameters to guide behavior
        strategy = interpret_strategy(self._approach.data)
        return strategy.create_tree()

# Optimize strategy using LangOptim
optimizer = LangOptim(llm=llm, params=strategy.param_set())
optimizer.step()  # LLM improves strategy text
```

## Next Steps

- **[State Machines](state-machines.md)** - More structured state-based behavior
- **[Process Framework](process-framework.md)** - Integrate behavior trees with processes
- **[Computational Graphs](computational-graphs.md)** - Compose behavior trees in DAGs
- **[Tutorial: Adaptive Behavior Trees](tutorial-adaptive-behavior-trees.md)** - Build self-modifying trees

---

Behavior trees in Dachi are powerful, composable, and adaptive - enabling you to build AI agents that can modify their own decision-making logic at runtime.
