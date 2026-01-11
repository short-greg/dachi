# Documentation Update Plan - 2025-12-16

**Status: IN PROGRESS - Major sections completed, core optimization docs remaining**

## Overview

The Dachi documentation contains critical errors that prevent code examples from running, as well as extensive documentation of features that don't exist in the production codebase. This plan addresses updating documentation to accurately reflect the current state of the framework.

**UPDATE 2025-12-17**: Significant progress made. Documentation now correctly positions Dachi as an ML framework using text as parameters. Core feature documentation created for behavior trees, processes, computational graphs, and criteria. Obsolete files removed. Critical optimization guide remains to be written.

## Critical Issues Identified

### 1. **Import Path Errors** (CRITICAL - Code Won't Run)

**Problem: `dachi.comm` module doesn't exist**
- Documentation uses: `from dachi.comm import Blackboard, Bulletin, AsyncDispatcher`
- Actual location:
  - `from dachi.act.comm import Blackboard, Bulletin`
  - `from dachi.proc import AsyncDispatcher`
- Affected files:
  - README.md (lines 19, 23, 83, 89)
  - docs/quick-start.md (lines 23, 142)
  - docs/communication-and-requests.md
  - docs/tutorial-multi-agent-communication.md
  - docs/architecture-in-practice.md

**Problem: `Msg` not in utils**
- Documentation uses: `from dachi.utils import Msg`
- Actual location: `from dachi.core import Msg`
- Verified: dachi/utils/__init__.py exports only: singleton, store, func, text
- Verified: dachi/core/__init__.py exports Msg from _base.py
- Affected files:
  - README.md (line 21, 123)
  - docs/quick-start.md (line 25, 144)
  - All tutorial files

### 2. **Non-Existent Adapter Classes** (CRITICAL - Major Feature Claims)

Documentation extensively describes AI adapter classes that don't exist in the main codebase:
- `OpenAIChat` - Not in dachi.proc
- `AnthropicChat` - Not in main codebase
- `LocalChat` - Not in main codebase
- `OpenAIResp` - Not in main codebase

**What Actually Exists:**
- `LangModel` - Abstract base class in dachi/proc/_ai.py (exported from dachi.proc)
- Experimental adapters in `/local/adapt/` (not production-ready)

Affected files:
- README.md (lines 20, 26, 122, 126)
- docs/quick-start.md (lines 24, 30, 166)
- docs/adapters.md (ENTIRE FILE - ~415 lines of non-existent API)
- All tutorial files with AI examples

### 3. **Missing Response Classes**

Documentation references:
- `Resp` and `RespDelta` - Not exported from dachi.core
- May exist internally but not in public API

### 4. **CLAUDE.md Inaccuracies**

Issues found:
- Line 24: "ShareableItem hierarchy: Param, Attr, Shared" - Actually it's `Runtime` not `Attr`
- Line 33-35: "AI Integration (dachi/adapt/)" with OpenAI adapter - adapters are experimental in /local/
- Lines 163-169: Describes `forecasting/` and `dashboard/` folders that don't exist in this repo (wrong project?)

## Implementation Plan

### Phase 1: Critical Import Fixes (All documentation files)
- Replace `from dachi.comm import Blackboard, Bulletin` → `from dachi.act.comm import Blackboard, Bulletin`
- Replace `from dachi.comm import AsyncDispatcher` → `from dachi.proc import AsyncDispatcher`
- Replace `from dachi.utils import Msg` → `from dachi.core import Msg`
- Verify all import statements in code examples

### Phase 2: Remove Non-Existent Adapter Documentation
- Remove all references to `OpenAIChat`, `AnthropicChat`, `LocalChat`, `OpenAIResp`
- Remove or heavily rewrite docs/adapters.md (entire file is about non-existent adapters)
- Update README.md to remove adapter examples
- Update all tutorials to remove AI adapter usage
- Document only `LangModel` abstract base class (what actually exists)
- Remove all `Resp` and `RespDelta` references (these were removed)

### Phase 3: CLAUDE.md Cleanup
- Fix ShareableItem terminology (Attr → Runtime)
- Remove entire "Operating Guide" section (lines 155-269) about forecasting/dashboard
- Update AI integration architecture claims (remove dachi/adapt/ references to experimental code)
- Verify all module locations match actual codebase

### Phase 4: README.md Updates
- Fix import paths in quick start example
- Update feature claims to match reality
- Remove or mark experimental features appropriately
- Ensure "Key Features" section is accurate

### Phase 5: Tutorial and Guide Updates
- Update all code examples with correct imports
- Remove or mark examples using non-existent adapters
- Ensure all examples can actually run

## Files Requiring Updates

### High Priority (Code Examples Won't Run)
- README.md - Main entry point with broken examples
- docs/quick-start.md - First user experience, must work
- docs/adapters.md - Entire file documents non-existent API
- docs/tutorial-simple-chat-agent.md
- docs/tutorial-multi-agent-communication.md

### Medium Priority (Inaccurate Claims)
- CLAUDE.md - Development guidance for contributors
- docs/core-architecture.md - Architecture claims
- docs/message-system.md - May reference non-existent classes
- docs/communication-and-requests.md - Import path issues

### Low Priority (General Cleanup)
- docs/usage-patterns.md
- docs/architecture-in-practice.md
- docs/design-pillars.md - Philosophy/concepts (less likely to have code issues)

## Implementation Strategy

1. Start with import fixes (Phase 1) - these are non-controversial and fix immediate breakage
2. Wait for user guidance on adapters before touching adapter-related documentation
3. Update CLAUDE.md after confirming what sections to keep/remove
4. Verify all changes by testing code examples if possible

## Success Criteria

- All code examples use correct import paths
- No documentation references non-existent classes without marking them as experimental/planned
- CLAUDE.md accurately describes the Dachi project structure
- Users can copy-paste examples from documentation and have them work
- Clear distinction between stable API and experimental features

## Detailed File-by-File Changes

### README.md
**Changes:**
1. Lines 19-44 (Quick Start section): Remove entire AI chat example using `OpenAIChat`
2. Lines 83-96 (Communication Layer): Fix import `from dachi.comm` → `from dachi.act.comm`
3. Lines 119-133 (AI Processing section): Remove entire section (uses non-existent `OpenAIChat`)
4. Line 150-153 (AI Integration): Remove claims about OpenAI adapters
5. Create new simplified Quick Start without AI components, or use only components that exist

### docs/quick-start.md
**Changes:**
1. Lines 23-25: Fix imports (`dachi.comm` → `dachi.act.comm`, `dachi.utils.Msg` → `dachi.core.Msg`)
2. Lines 24-86: Remove or rewrite entire "Hello World" example (uses `OpenAIChat`)
3. Lines 136-328: Remove or rewrite "Smart Task Processor" (heavily uses `OpenAIChat`)
4. Rewrite with examples using only available components (Blackboard, Bulletin, AsyncDispatcher, Task, Process)

### docs/adapters.md
**Decision:** This entire file (415 lines) documents non-existent adapter classes.
**Options:**
- Option A: Delete the file entirely
- Option B: Rewrite to document `LangModel` abstract base class and how to create custom adapters
**Recommendation:** Option B - keep the file but rewrite to show the extensibility pattern

### docs/tutorial-simple-chat-agent.md
**Changes:**
1. Fix all import paths
2. Remove AI adapter usage
3. Rewrite to focus on communication patterns without AI, or show Process extension pattern

### docs/tutorial-multi-agent-communication.md
**Changes:**
1. Fix imports: `dachi.comm` → `dachi.act.comm`
2. Remove AI adapter references
3. Focus on Bulletin/Blackboard communication patterns

### docs/communication-and-requests.md
**Changes:**
1. Fix all imports: `dachi.comm` → `dachi.act.comm`
2. Remove `Resp`/`RespDelta` references
3. Verify AsyncDispatcher examples are accurate

### docs/message-system.md
**Changes:**
1. Fix imports: `dachi.utils.Msg` → `dachi.core.Msg`
2. Remove all `Resp` and `RespDelta` references
3. Document only `Msg` class and its actual usage

### docs/core-architecture.md
**Changes:**
1. Verify all module locations are correct
2. Remove references to non-existent adapter classes
3. Update communication layer location (dachi.act.comm not dachi.comm)

### docs/architecture-in-practice.md
**Changes:**
1. Fix all import paths
2. Remove AI adapter examples
3. Update to show patterns with actual available components

### docs/usage-patterns.md
**Changes:**
1. Fix imports throughout
2. Remove adapter usage patterns
3. Keep patterns for Blackboard, Bulletin, AsyncDispatcher, Task, Process

### CLAUDE.md
**Changes:**
1. Line 24: "Param, Attr, Shared" → "Param, Runtime, Shared"
2. Lines 33-35: Remove or update "AI Integration (dachi/adapt/)" section
3. Lines 155-269: DELETE entire "Operating Guide" section about forecasting/dashboard
4. Update module organization to reflect actual structure
5. Update test commands if needed

## Implementation Steps

1. **Phase 1: Critical Import Fixes** (1-2 hours)
   - Search and replace across all .md files in docs/ and root
   - `from dachi.comm import` → `from dachi.act.comm import`
   - `from dachi.utils import Msg` → `from dachi.core import Msg`

2. **Phase 2: Remove Adapter References** (2-3 hours)
   - Remove all code examples using OpenAIChat, AnthropicChat, LocalChat, OpenAIResp
   - Rewrite docs/adapters.md to document LangModel extension pattern
   - Update README.md Quick Start to use only available components

3. **Phase 3: Remove Resp/RespDelta** (1 hour)
   - Search for all Resp/RespDelta references
   - Remove or rewrite sections that depend on these classes

4. **Phase 4: CLAUDE.md Cleanup** (30 min)
   - Fix terminology
   - Remove forecasting/dashboard section
   - Update architecture descriptions

5. **Phase 5: Rewrite Examples** (3-4 hours)
   - Create working examples using only: Blackboard, Bulletin, AsyncDispatcher, Task, Process, LangModel
   - Update tutorials to show real, working patterns
   - Focus on framework capabilities without external AI services

6. **Phase 6: Verification** (1 hour)
   - Check all import statements
   - Verify no references to removed classes
   - Ensure consistency across all docs

## Total Estimated Time: 8-11 hours (Originally) → ~34.5 hours (Revised for Recreation)

## PROGRESS UPDATE - 2025-12-17

### ✅ COMPLETED WORK (18+ hours)

**New Documentation Created (6 files):**
1. ✅ **docs/quick-start.md** - Complete rewrite with behavior trees, DataFlow, and text parameters
   - Example 1: Dynamic behavior trees (order fulfillment scenario)
   - Example 2: Computational graphs (DataFlow with (5+3)*2=16)
   - Example 3: Text parameters with correct PrivateParam syntax
   - Clear "What's Next" navigation sections

2. ✅ **README.md** - Complete rewrite emphasizing ML framework
   - Title: "Dachi - ML Framework for Adaptive AI Systems"
   - Opening: Text as parameters instead of numerical parameters
   - 5 key capabilities with code examples
   - Architecture overview: Core, Action, Processing, Instruction layers
   - Comparison table: Traditional ML vs Dachi
   - De-emphasized LangModel (mentioned as integration option)

3. ✅ **docs/behavior-trees-and-coordination.md** - Comprehensive BT guide
   - TaskStatus enum and all properties
   - Creating custom Actions with correct Task base class
   - Composite tasks: SequenceTask, SelectorTask, ParallelTask (correct names)
   - Decorators: Not, AsLongAs, Until, PreemptCond
   - Dynamic behavior tree creation patterns
   - Integration with Modules and DataFlow

4. ✅ **docs/process-framework.md** - Complete process documentation
   - 4 execution modes: Process, AsyncProcess, StreamProcess, AsyncStreamProcess
   - Helper functions: forward/aforward/stream/astream
   - Multi-interface processes
   - Process composition patterns (sequential, DataFlow, wrappers)
   - Practical examples: data pipelines, async APIs, streaming
   - Best practices and common patterns

5. ✅ **docs/computational-graphs.md** - DataFlow complete guide
   - Core concepts: V (Variable) nodes, Process nodes, Ref objects
   - Building graphs: add_inp(), link(), set_out()
   - Advanced patterns: diamond dependency, branching/merging, runtime overrides
   - Parallel execution with asyncio
   - Node references and indexing
   - Graph manipulation: sub-graphs, replacing nodes
   - Integration with behavior trees and Module system

6. ✅ **docs/criterion-system.md** - ResponseSpec and evaluation
   - ResponseSpec base class and auto-generated schemas
   - RespField descriptors: TextField, BoolField, BoundInt, BoundFloat, ListField, DictField
   - All built-in criteria: PassFail, Likert, NumericalRating, Checklist, HolisticRubric, AnalyticRubric, Narrative, Comparative
   - Creating custom criteria (CodeReviewCriterion example)
   - LangCritic integration
   - Batch evaluation support

**Files Deleted (Obsolete/Incorrect):**
- ✅ docs/adapters.md (415 lines documenting non-existent OpenAIChat/AnthropicChat/LocalChat)
- ✅ docs/tutorial-simple-chat-agent.md (chat bot tutorial with non-existent classes)
- ✅ docs/tutorial-multi-agent-communication.md (multi-agent chat, wrong framing)
- ✅ docs/message-system.md (user confirmed: "there isn't really a message system anymore")

**Files Updated:**
- ✅ **docs/communication-and-requests.md** - Fixed all imports
  - `from dachi.comm` → `from dachi.act.comm` (Bulletin, Blackboard, Buffer, AsyncDispatcher)
  - Removed `OpenAIChat` reference, replaced with generic `my_llm_adapter`
  - All code examples now use correct paths

- ✅ **CLAUDE.md** - Already updated (by linter)
  - Project overview mentions text parameters and Bayesian updating
  - Architecture sections updated with ParamSet, LangOptim, LangCritic
  - Removed forecasting/dashboard sections (user confirmed wrong project)

### 🔄 REMAINING WORK (Estimated 12.5 hours)

**Priority 1: Critical Missing Documentation (4.5 hours)**

1. **docs/optimization-guide.md** (NOT STARTED - 3 hours)
   - **THIS IS THE CORE INNOVATION - HIGHEST PRIORITY**
   - Complete LangOptim workflow documentation
   - Text parameter optimization via Bayesian updating
   - Module → ParamSet → Criterion → LangCritic → LangOptim
   - Text parameter patterns and best practices
   - Examples: prompt optimization, strategy tuning
   - Integration with Module system

2. **docs/langmodel-adapters.md** (NOT STARTED - 1.5 hours)
   - LangModel abstract base class specification
   - Four methods: forward/aforward/stream/astream
   - Return tuple format: `(str, List[Inp], Any)`
   - How to implement adapters (de-emphasized, brief)
   - Note that Msg/Inp exist for API compatibility

**Priority 2: Update Existing Files (8 hours)**

3. **docs/core-architecture.md** (NEEDS UPDATE - 2 hours)
   - Add "Text Parameters" section explaining Dachi's innovation
   - Add LangOptim/LangCritic/ParamSet architecture
   - Fix any remaining import paths
   - Update module hierarchy diagram
   - Emphasize Param/Runtime/Shared hierarchy

4. **docs/design-pillars.md** (NEEDS REWRITE - 2 hours)
   - Remove non-existent adapter references (lines 92-99: OpenAIChat, AnthropicChat, LocalLLM)
   - Add text-as-parameters innovation section
   - Emphasize Bayesian optimization vs gradient descent
   - Update examples with working code
   - Keep philosophical aspects, fix technical claims

5. **docs/architecture-in-practice.md** (NEEDS REWRITE - 2.5 hours)
   - Fix all import paths: `from dachi.comm` → `from dachi.act.comm`
   - Remove all `OpenAIChat` references (line 31, etc.)
   - Remove `Msg/Resp` architecture claims (line 10-11)
   - Rewrite examples with: behavior trees, DataFlow, text parameters
   - Only include working code examples

6. **docs/usage-patterns.md** (NEEDS UPDATE - 1 hour)
   - Fix imports throughout: `from dachi.comm` → `from dachi.act.comm`
   - Remove `OpenAIChat` references (lines 229, 236)
   - Keep the pattern structures, fix the code
   - Update AsyncDispatcher examples

7. **docs/serialization-and-state.md** (MINOR UPDATE - 0.5 hours)
   - Appears mostly correct already
   - Add text parameter context where relevant
   - Emphasize spec/state pattern for optimization workflows
   - Verify all examples are accurate

### 📊 DOCUMENTATION STATUS MATRIX

| File | Status | Action | Est. Hours | Priority |
|------|--------|--------|------------|----------|
| README.md | ✅ Complete | None | - | - |
| docs/quick-start.md | ✅ Complete | None | - | - |
| docs/behavior-trees-and-coordination.md | ✅ Complete | None | - | - |
| docs/process-framework.md | ✅ Complete | None | - | - |
| docs/computational-graphs.md | ✅ Complete | None | - | - |
| docs/criterion-system.md | ✅ Complete | None | - | - |
| docs/communication-and-requests.md | ✅ Updated | None | - | - |
| CLAUDE.md | ✅ Updated | None | - | - |
| docs/optimization-guide.md | ❌ Missing | CREATE | 3.0 | P1 |
| docs/langmodel-adapters.md | ❌ Missing | CREATE | 1.5 | P1 |
| docs/core-architecture.md | 🔄 Outdated | UPDATE | 2.0 | P2 |
| docs/design-pillars.md | 🔄 Outdated | REWRITE | 2.0 | P2 |
| docs/architecture-in-practice.md | 🔄 Outdated | REWRITE | 2.5 | P2 |
| docs/usage-patterns.md | 🔄 Outdated | UPDATE | 1.0 | P2 |
| docs/serialization-and-state.md | 🔄 Minor | UPDATE | 0.5 | P2 |

**DELETED (Obsolete):**
- ~~docs/adapters.md~~
- ~~docs/tutorial-simple-chat-agent.md~~
- ~~docs/tutorial-multi-agent-communication.md~~
- ~~docs/message-system.md~~

### 📈 PROGRESS METRICS

- **Total Documentation Files**: 12 (after cleanup)
- **Completed**: 8 files (67%)
- **Remaining**: 7 files (33% - 2 new, 5 updates)
- **Hours Invested**: ~18 hours
- **Hours Remaining**: ~12.5 hours
- **Total Project**: ~30.5 hours

### 🎯 KEY ACHIEVEMENTS

1. ✅ Documentation now positions Dachi as ML framework (not LLM wrapper)
2. ✅ Text-as-parameters concept explained in README and quick-start
3. ✅ All new documentation uses correct import paths
4. ✅ No references to non-existent classes in new files
5. ✅ Working code examples for behavior trees, DataFlow, and criteria
6. ✅ Obsolete/incorrect documentation removed
7. ⏳ LangOptim workflow NOT YET documented (critical gap)

### 🚨 CRITICAL GAP

**docs/optimization-guide.md is NOT STARTED** - This documents Dachi's core innovation (text parameter optimization via LangOptim). This is the highest priority remaining task.

## Next Steps

1. **IMMEDIATE**: Create docs/optimization-guide.md (3 hours) - THE CORE FEATURE
2. Create docs/langmodel-adapters.md (1.5 hours) - Complete the core docs
3. Update docs/core-architecture.md (2 hours) - Add text parameters section
4. Rewrite docs/design-pillars.md (2 hours) - Fix philosophy/examples
5. Rewrite docs/architecture-in-practice.md (2.5 hours) - Working examples only
6. Update docs/usage-patterns.md (1 hour) - Fix imports
7. Minor update docs/serialization-and-state.md (0.5 hours)
