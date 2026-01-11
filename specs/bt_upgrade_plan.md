# Dachi BT – Builder-First Redesign (Comparison, Rationale, and Examples)

# Plan to upgrade Dachi behavior trees

## Overview

We are updating our behavior tree design so trees are easy to write, safe to run, and straightforward for an LLM to generate. The goals are simple:

* Avoid global variables
* Keep data scoped to the subtree that produced it
* Make dataflow explicit right next to the node that needs it (inline binding)
* Let a single place decide what becomes visible above the tree
* Expose `tick()` inputs/outputs in the schema so an LLM can wire nodes correctly

### Final outcome (what it looks like)

```python
# new style (plain, readable)
tree = Sequence([
  Plan(tag="planA"),                 # produces planA.goal
  Bind(Move(tag="moveA"),            # bind inputs next to the consumer
       target="planA.goal",          # tag alias (or use index path "0.goal")
       attempts=2),                   # constant
])

app = Root(
  tree,
  export="all",                      # what to bubble upward ("all" | "none" | {keys} | [globs] | callable)
  namespace=None                      # optional prefix for all promoted keys
)
```

You can also bind by index path (canonical addressing):

```python
# index addressing is the canonical machine path
tree = Sequence([
  Plan(),                             # index 0
  Bind(Move(), target="0.goal"),     # consume by index path
])
```

---

## Naming decision for ports (final)

Use **non-dunder nested classes**:

* `class inputs:` declares keyword-only parameters for `tick()` with types and defaults
* `class outputs:` declares output keys and their types (structure when applicable)

> We avoid `__inputs__`/`__outputs__` because Python name-mangles dunders, which complicates reflection and inheritance.

**Examples**

```python
from typing import TypedDict

class Plan(Action):
  class outputs(TypedDict):
    goal: tuple[float, float, float]
  async def tick(self) -> tuple[Status, outputs]:
    return Status.SUCCESS, {"goal": (1.0, 2.0, 0.0)}

class Move(Action):
  class inputs:
    target: tuple[float, float, float]
    attempts: int = 2
  class outputs:
    arrived: bool
  async def tick(self, *, target, attempts):
    ...
    return Status.SUCCESS, {"arrived": True}
```

---

## 1) Change: Add **inputs** to the `tick()` method

### Current state

* The spec only tells the LLM about constructor fields.
* `tick()` inputs are implicit and not discoverable from the schema, so the LLM cannot know what to bind.

### What the change will be

* Each node declares `tick()` keyword-only inputs via `class inputs:` (types + defaults). The LLM and the builder treat these as the inputs to the node’s `tick()`.

### How it will be done

* Keep user code as above.
* During leaf class creation (`__init_subclass__`), parse and cache inputs on the class; during spec build, emit them to JSON Schema as `bt_ports.inputs`.

**Usage**

```python
class Move(Action):
  class inputs:
    target: tuple[float, float, float]
    attempts: int = 2
  async def tick(self, *, target, attempts): ...
```

---

## 2) Change: Add **outputs** to the `tick()` method (declared on the class)

### Current state

* Node outputs are returned but not described to the LLM, so it cannot route values or pick subfields.

### What the change will be

* Each node declares output names and types via `class outputs:` (prefer `TypedDict` for structured shapes). The values are dynamic per tick; the **contract** is static.

### How it will be done

* During leaf class creation, parse and cache outputs on the class; during spec build, emit them to JSON Schema as `bt_ports.outputs` (including structure where known).

**Usage**

```python
from typing import TypedDict

class Sense(Action):
  class outputs(TypedDict):
    pose: tuple[float, float, float]  # (x, y, theta)
  async def tick(self) -> tuple[Status, outputs]: ...
```

---

## 3) Change: Composites/Decorators take a **Context** and route inputs

### Current state

* Composites don’t receive a structured context, and input routing is ad hoc.

### What the change will be

* `Sequence`, `Selector`, and decorators (including `Bind`) accept a `ctx` in `tick(ctx)` and are responsible for:

  * reading sibling-local outputs (produced earlier in the same tick)
  * reading from the scoped context (parent → root)
  * resolving the child’s `class inputs:` into keyword args
  * calling the child with `await child.tick(**resolved_kwargs)`

### How it will be done

* Add `tick(ctx)` on all composites/decorators.
* Maintain a small per-tick local map inside each composite for sibling outputs.
* **Resolution order** per input: (1) sibling locals → (2) `ctx` → (3) default → else unresolved → composite policy (see §5).

**Usage**

```python
Sequence([
  Plan(tag="plan"),                    # produces plan.goal
  Bind(Move(tag="move"), target="plan.goal"),
]).tick(ctx)
```

---

## 4) Change: `Bind` is a Decorator that adapts **inputs only**, resolved dynamically

### Current state

* Binding is distant from the consumer or implicit; not resilient to timing.

### What the change will be

* `Bind(node, **inputs)` wraps a node and adapts its **inputs** at **tick-time**.
* Allowed binding values:

  * string key (tag or index path): `target="plan.goal"` or `"0.goal"`
  * constant: `attempts=2`
  * structured subfields: `"plan.pose.2"` or `"plan.goal.x"`
* Outputs are not renamed or adapted by `Bind`; they pass through unchanged.

### How it will be done

* `Bind.tick(ctx)` evaluates its bound expressions first; for any param not provided, the composite’s resolution (locals → ctx → default) fills in.

**Usage**

```python
Sequence([
  Sense(tag="sense"),                  # sense.pose -> (x,y,theta)
  Bind(Rotate(tag="turn"), theta="sense.pose.2"),
])
```

---

## 5) Change: Policy for unresolved inputs (per-composite, simple)

### Current state

* Unresolved inputs cause unpredictable behavior.

### What the change will be

* Composites expose a **single small policy**:

  * `missing="fail"` (default): unresolved required input → return `FAILURE`
  * `missing="defer"`: unresolved → return `RUNNING` (retry next tick)

### How it will be done

* Add optional `missing` parameter to composites (e.g., `Sequence(..., missing="fail")`). Root does **not** need `missing`.

**Usage**

```python
Sequence([
  MaybePlan(tag="plan"),               # may not produce plan.goal yet
  Bind(Move(), target="plan.goal"),
], missing="defer")
```

---

## 6) Change: Root owns export and namespacing (kept minimal)

### Current state

* Export control scattered across the tree.

### What the change will be

* `Root` is the only place that:

  * creates a per-tick scoped snapshot
  * **promotes** outputs on SUCCESS based on its export rule
  * applies an optional namespace prefix

### How it will be done

* `Root(tree, export=..., namespace=...)`

  * `export`: "all" | "none" | set of keys | list of globs | callable `(key, value, path) -> (new_key, export_bool)`
  * `namespace`: optional prefix for promoted keys
* On RUNNING/FAILURE, Root exports nothing for that tick.
* Collisions on final keys are **dev errors** for now (no merge knob). If a real use case appears (e.g., Parallel with combination), we’ll add a local policy on that composite.

**Usage**

```python
app = Root(tree, export="all")
app = Root(tree, export={"plan.goal", "nav.eta"})
app = Root(tree, export=["nav.*", "plan.*"], namespace="mission")
```

---

## 7) Change: `__init_subclass__` caches ports on leaves (Action/Conditional)

### Current state

* Ports are repeated or discovered late; schema and runtime can drift.

### What the change will be

* When a leaf class is defined, `__init_subclass__` reflects `class inputs:` and `class outputs:` once and stores them on the class for both:

  * runtime resolution (composites/decorators read ports quickly)
  * schema generation (`bt_ports` extras) so the LLM can auto-bind

### How it will be done

* On `Action`/`Conditional` base, implement `__init_subclass__` to:

  * parse `class inputs:` annotations + defaults
  * parse `class outputs:` (TypedDict preferred; otherwise annotated class)
  * cache `__ports__ = {"inputs": {...}, "outputs": {...}}` on the class
* The spec builder reads this cache to populate `bt_ports` in JSON Schema.

**Usage (authoring a leaf)**

```python
class Move(Action):
  class inputs:
    target: tuple[float, float, float]
    attempts: int = 2
  class outputs:
    arrived: bool
  async def tick(self, *, target, attempts): ...
```

---

## End-to-end examples

### A) Simple dock-and-move

```python
tree = Sequence([
  Plan(tag="plan"),                    # plan.goal
  Bind(Move(tag="move"), target="plan.goal", attempts=2),
])
app = Root(tree, export="all")
```

### B) Structured subfield and tag/index mix

```python
tree = Sequence([
  Sense(tag="sense"),                  # sense.pose -> (x,y,theta)
  Bind(Rotate(tag="turn"), theta="sense.pose.2"),
  ComputeETA(tag="nav"),               # nav.eta
])
app = Root(tree, export=["plan.*", "nav.*"], namespace="mission")
```

### C) Deferred missing inputs

```python
tree = Sequence([
  MaybePlan(tag="plan"),               # may need multiple ticks
  Bind(Move(tag="move"), target="plan.goal"),
], missing="defer")
app = Root(tree, export="all")
```

---

## Implementation steps (lean)

1. **`__init_subclass__` on leaves**: capture `class inputs:` and `class outputs:`; cache on class; update spec builder to emit `bt_ports` extras.
2. **Composites/decorators with `ctx`**: add `tick(ctx)` and local sibling store; implement resolution order (locals → ctx → default → policy).
3. **`Bind` decorator**: inputs-only adapter; supports key/constant/subfield strings; resolves at tick-time before composite fill-ins.
4. **Root**: export + namespace only; strict duplicate detection (raise in dev).
5. **Tests**: (a) ports in JSON Schema, (b) inline binds (tag/index/subfield), (c) composite resolution and `missing` policy, (d) Root export rules and namespacing.

---

## Success criteria

* Trees remain concise; binding is next to the consumer.
* Only Root controls visibility; no per-subtree export knobs.
* The LLM can generate correct trees (including structured binds) using `bt_ports`.
* Tags and indices can be used interchangeably; canonical addressing stays stable.

---

## Implementation Progress and Decisions

### ✅ Completed (Phase 1: Foundation)

#### Context/Scope System
- **Files**: `dachi/core/_scope.py`, `dachi/core/__init__.py`
- **Implementation**: Complete Scope/Ctx system with hierarchical data storage
- **Key Decisions**:
  - `Scope(dict)` stores all data with tuple keys for hierarchical access
  - `Ctx` is lightweight proxy that knows its position in the hierarchy  
  - `scope.ctx()` creates **root context** (no arguments = root level)
  - `scope.ctx(0, 1, 2)` creates child contexts with index paths
  - Path resolution supports both index ("0.1.goal") and tag ("plan.goal") patterns
  - Tag aliases stored in `scope.aliases` dict for name-to-path mapping

#### Port System for Leaf Nodes  
- **Files**: `dachi/act/_bt/_core.py`, `tests/act/test_leafs.py`
- **Implementation**: Complete port declaration and processing system
- **Key Decisions**:
  - Use `class inputs:` and `class outputs:` (non-dunder nested classes)
  - `__init_subclass__` automatically processes port declarations on class creation
  - `_process_ports()` extracts annotations and defaults from port classes
  - `__ports__` class attribute caches processed port information
  - **Critical Fix**: Added default empty `inputs` and `outputs` classes to base `Leaf` class
  - Port declarations are **optional** - classes without ports get empty port dictionaries
  - All 134 existing behavior tree tests continue to pass

#### Testing Infrastructure
- **Files**: `CLAUDE.md`, `tests/act/test_leafs.py`
- **Implementation**: Established testing conventions and comprehensive port system tests
- **Decisions**:
  - Test class naming: `TestClassName` and `TestClassNameMethodName`
  - Test method naming: `test_<method>_<behavior>_<condition>`
  - Port tests integrated into `test_leafs.py` (not separate file)
  - 7 comprehensive tests cover port extraction, inheritance, and integration

### ✅ Completed (Phase 2: Context-Aware Composites)

#### Context-Aware Composites Implementation
- **Files**: `dachi/act/_bt/_serial.py`, `dachi/act/_bt/_decorators.py`, `dachi/act/_bt/_core.py`
- **Implementation**: Complete context-aware behavior tree system
- **Key Decisions**:
  - Added `ctx_tick()` method to `Leaf` base class for unified input resolution handling
  - All composites (Sequence, Selector, Decorators) now use `ctx_tick()` for leaf tasks
  - Missing required inputs automatically fail the child task (not the composite)
  - `ctx_tick()` catches KeyError from `build_inputs()` and calls `task.fail()` 
  - Added `fail()` and `succeed()` methods to Task base class for direct status setting

#### Comprehensive Test Coverage
- **Files**: `tests/act/test_serial.py`, `tests/act/test_decorators.py`, `tests/act/test_leafs.py`
- **Status**: All tests passing (52/52 serial, 14/14 decorators, 25/25 leafs, 26/26 scope)
- **Test Classes Added**:
  - `TestSequenceWithContext` (8 tests): Child context creation, input resolution, output storage
  - `TestSelectorWithContext` (4 tests): Context propagation, success isolation, failure handling
  - `TestContextUpdate` (3 tests): Ctx.update() functionality and dict-like behavior
  - `TestCompositeInputResolution` (7 tests): Cross-cutting input resolution scenarios
  
#### Input Resolution System  
- **Implementation**: Complete input resolution with graceful failure handling
- **Key Features**:
  - Leaf classes can declare required inputs via `class inputs:` with type hints
  - Missing required inputs cause individual task failure (not composite failure)
  - Optional inputs fall back to defaults when missing from context
  - Context data accessible at correct hierarchical paths
  - Sibling output storage and retrieval working correctly

#### Context Flow Architecture
- **Implementation**: Hierarchical context system with proper data isolation
- **Key Features**:
  - Composites create child contexts at indexed paths: `ctx.child(0)`, `ctx.child(1)`, etc.
  - Leaf tasks receive parent context for input resolution
  - Outputs stored at child context paths: `scope[(0, "result")]`, `scope[(1, "value")]`
  - Context isolation between attempts in selectors
  - Proper context forwarding through decorator chains

### 📋 Pending (Future Phases)

#### Parallel Class Context Support
- **Files**: `dachi/act/_bt/_parallel.py`
- **Status**: Needs update to context-aware pattern
- **Requirements**:
  - Update `Multi.tick(ctx)` to accept context parameter
  - Use `ctx_tick()` for leaf tasks, `task.tick(ctx=child_ctx)` for composites
  - Create child contexts for each parallel task at indexed paths
  - Handle input resolution failures gracefully (individual task failures)
  - Update `run()` method to pass context through to child tasks

#### Root/BT Class Context Support  
- **Files**: `dachi/act/_bt/_roots.py`
- **Status**: Needs update to context-aware pattern
- **Requirements**:
  - Update `BT.tick(ctx)` to accept context parameter
  - Forward context to adapted root task
  - Handle context creation and scope management at tree root level
  - Integration with future export control system

#### Bind Decorator Implementation
- **Status**: Ready for implementation (infrastructure complete)
- **Target**: Implement `Bind` decorator for inline input binding  
- **Requirements**:
  - Support string expressions: tags, indices, constants, subfields
  - Resolve bindings at tick-time before composite resolution
  - Inputs-only adapter (outputs pass through unchanged)
  - Integration with existing `ctx_tick()` pattern

#### Enhanced Input Resolution  
- **Status**: Core system complete, enhancements possible
- **Potential Improvements**:
  - Sibling data flow between children (producer → consumer in same composite)
  - Resolution priority: sibling outputs → context → defaults
  - Path resolution with dot notation ("plan.goal.x", "0.sensors.pose.x")
  - Tag alias system integration with path resolution

#### Root Export Control
- **Status**: Not yet started
- **Target**: Implement export rules and namespacing in Root node
- **Requirements**:
  - Handle promotion of successful outputs based on export policy
  - Support export patterns: "all", "none", key sets, globs, callables
  - Optional namespace prefixing
  - Collision detection for duplicate export keys

### 🔧 Key Technical Decisions Made

1. **Context Naming**: Changed from "Context" to "Scope/Ctx" to avoid naming conflicts
2. **Root Context**: `scope.ctx()` with no arguments creates root context (not `scope.ctx(0)`)
3. **Port Optionality**: Port declarations are optional for backward compatibility
4. **Base Class Ports**: Added empty default port classes to prevent `__init_subclass__` failures
5. **Test Organization**: Port system tests integrated into existing leaf test file
6. **Implementation Order**: Context/Scope first, then ports, then composites, then decorators

### 🚧 Known Challenges

1. **Backward Compatibility**: Need to maintain existing `tick()` signatures while adding context support
2. **Resolution Complexity**: Implementing proper resolution order for composite input binding
3. **Error Handling**: Graceful handling of unresolved inputs with configurable policies
4. **Testing Strategy**: Ensuring comprehensive end-to-end testing of context flow

### 📝 Next Steps

1. **Immediate**: Implement context-aware tests for Sequence/Selector composites
2. **Implement**: Update Sequence/Selector `tick(ctx)` methods with resolution logic  
3. **Test**: Verify context forwarding and sibling output handling
4. **Extend**: Update decorators to accept and forward context
5. **Integrate**: End-to-end testing with full context + port + composite integration
