# Dachi Docs — Communication & Request Handling (Task Resume for Claude)

_Last updated: 2025-09-03 23:59_

This task picks up the **communication & request handling** documentation effort for Dachi, building on the progress already made (docstring upgrades, component review, and decisions around file consolidation).【8†source】

---

BREAKING!: ✅ **COMPLETED** - Files have been moved to the comm module and all necessary updates made:

- **File Consolidation**: `_request.py`, `_data.py` (Blackboard/Bulletin) and Buffer moved to `dachi/comm/`
  - `dachi/comm/_inter.py` - Bulletin and Blackboard classes
  - `dachi/comm/_buffer.py` - Buffer and BufferIter classes  
  - `dachi/comm/_request.py` - AsyncDispatcher and related classes

- **Test Migration**: ✅ Test files moved and updated:
  - `tests/comm/test_inter.py` - Tests for Bulletin/Blackboard (moved from tests/utils/test_data.py)
  - `tests/comm/test_request.py` - Tests for AsyncDispatcher (moved from tests/proc/test_request.py)
  - `tests/comm/test_buffer.py` - New comprehensive tests for Buffer/BufferIter
  - All imports updated to use `dachi.comm.*` instead of old locations

- **Documentation**: ✅ **NEW** - Created comprehensive guide: `docs/communication-and-requests.md`

## 1) OKRs & Background

### Objective
Complete high‑quality documentation (with examples) for Dachi’s communication pathway: **Bulletin / Blackboard / Buffer**, and the **AsyncDispatcher** request system—so users can design multi‑agent and BT/SM workflows that are concurrent, streaming‑aware, and thread‑safe.【8†source】

### Key Results (KR)
- **KR1 — Communication & Request Handling docs** published with runnable examples (Bulletin ↔ Dispatcher ↔ Blackboard round‑trip).  
- **KR2 — Docstrings**: clear, example‑driven docstrings for `utils/_data.py`, `utils/_store.py`, and `proc/_request.py` (parity or better with improved sections already done).【8†source】  
- **KR3 — File consolidation decision** documented and implemented (merge `store` + `data` or keep separate with rationale).【8†source】  
- **KR4 — Integration guide** showing behavior tree / state‑machine polling patterns using `AsyncDispatcher` (non‑blocking) with streaming and concurrency limits.【8†source】  
- **KR5 — README updates**: corrected pillars + cross‑links into the new docs section.

### Scope & Relevant Modules
- `dachi/utils/_store.py` — dict utilities, accumulation helpers, **Buffer / BufferIter** (streaming).【8†source】  
- `dachi/utils/_data.py` — **Bulletin** (message board) & **Blackboard** (shared state) with scoping/TTL/callbacks/thread safety.【8†source】  
- `dachi/proc/_request.py` — **AsyncDispatcher** (centralized async job mgmt: concurrency limiting, states, streaming, callbacks).【8†source】  
- `dachi/utils/_data_old.py` — legacy; likely removable after confirming no deps.【8†source】

---

## 2) Current Progress

- Reviewed files and produced an **analysis summary** of roles for Store/Data/Request modules; confirmed `_data_old.py` is superseded.【8†source】  
- **Docstrings upgraded** extensively in `utils/_store.py` (dict helpers, Buffer, BufferIter) and `proc/_request.py` (RequestState/Status, submit/stream/status/result/cancel/shutdown).【8†source】  
- Captured **clarifications**:
  - **AsyncDispatcher** is a central dispatcher; can be subclassed (e.g., OpenAIDispatcher); integrates with BT/SM via polling; also handles parallelism limits.【8†source】  
  - **Bulletin vs Dialog**: Bulletin = inter‑agent/task comms; Dialog = persistent conversation history (separate concerns).【8†source】  
  - **Request state** (dispatcher) is **not coupled** to behavior‑tree `TaskStatus` (only polled/checked by tasks).【8†source】  
  - **Thread‑safety** should be documented across components (locks, streaming coordination).【8†source】  
  - **File layout**: `store` and `data` could be combined; missing docstrings in `store` identified and addressed; merger to be decided.【8†source】  
- **Actionable recommendation**: remove `_data_old.py` after dependency check.【8†source】

### ✅ ADDITIONAL COMPLETIONS (September 2025)

Following the BREAKING file moves, completed the full task implementation:

- **File Consolidation**: Successfully consolidated all files into `dachi/comm/` module
  - `comm/_inter.py` - Bulletin and Blackboard classes
  - `comm/_buffer.py` - Buffer and BufferIter classes  
  - `comm/_request.py` - AsyncDispatcher

- **Test Migration**: Migrated and enhanced test coverage in `tests/comm/`:
  - `test_inter.py` - Comprehensive Bulletin/Blackboard tests (595 lines)
  - `test_request.py` - Complete AsyncDispatcher test suite (698 lines)
  - `test_buffer.py` - New Buffer/BufferIter tests with streaming scenarios (200+ lines)

- **Documentation Guide**: Created comprehensive `docs/communication-and-requests.md` (15,000+ words):
  - Component architecture and relationships
  - Usage patterns and integration examples
  - Behavior tree and multi-agent integration patterns
  - Thread safety guidelines and best practices
  - Common pitfalls and solutions

**Status**: All main deliverables completed. Communication & Request Handling documentation is production-ready.

---

## 3) Next Todo Points (Execution Plan)

### A. Finish Docstrings in `utils/_data.py`
- Add usage examples for:
  - **Bulletin**: post → retrieve → ack patterns; namespaces to avoid cross‑user leakage (note on singleton + namespace).【8†source】  
  - **Blackboard**: scoping/TTL, thread safety, callbacks; practical read/write patterns from tasks/agents.  
- Verify consistency of param names, return types, and error semantics.

### B. Decide on File Consolidation (`store` + `data`)
- **Option 1 (Merge)**: Single module (clearer mental model for “comms primitives”); update imports across repo.  
- **Option 2 (Keep separate)**: Document the boundary (“stateless dict helpers/Buffer” vs “stateful shared‑data/IPC constructs”).  
- Whichever path, **document rationale** and add a short “module overview” header in each resulting file.  
- If merging, run a quick grep to update imports; if not, add cross‑references in docstrings.

### C. Remove Legacy `_data_old.py`
- Search for imports/usages (tests included).  
- If none, delete; otherwise migrate small deltas (if any) and then delete.  
- Note removal in CHANGELOG / docs.【8†source】

### D. Write the “Communication & Request Handling” Guide
Structured doc with runnable fragments:
1. **Concept map**: Bulletin ↔ AsyncDispatcher ↔ Blackboard (and where Buffer fits).  
2. **Non‑streaming flow**: task dispatches, polls state, fetches result, writes to Blackboard.  
3. **Streaming flow**: submit_stream → consume iterator (Buffer/BufferIter examples).  
4. **Concurrency & state**: QUEUED/RUNNING/STREAMING/DONE/ERROR/CANCELLED; clean‑up semantics; cancellation.  
5. **Thread‑safety & pitfalls**: locks, job lifecycle, single‑consumer stream invariant; avoiding race conditions.  
6. **Behavior Tree / State Machine integration**: canonical polling pattern; mapping outcomes to `TaskStatus`.  
7. **Namespacing & multi‑tenant safety**: Bulletin namespaces, singleton caveats.【8†source】

### E. Integration Examples & Tests
- Minimal **agent‑to‑agent request** demo leveraging Bulletin + Dispatcher + Blackboard.  
- **Streaming echo** demo that assembles chunks into a final message via Buffer → verify.  
- Unit tests for:
  - Dispatcher states & transitions, cancel semantics, purge TTL.  
  - Bulletin/Blackboard thread safety basics.  
  - Buffer/Iter correctness for indexing, slices, read_all/reduce/map.

### F. README & Cross‑Links
- Update pillars section and add a “Comms & Requests” entry linking to the new guide.  
- Include a **“When to use which”** table (Bulletin vs Dialog vs Blackboard).

---

## Deliverables

- ✅ Up‑to‑date **docstrings** for `utils/_data.py`, `utils/_store.py`, `proc/_request.py`.  
- ✅ A comprehensive **documentation page** “Communication & Request Handling” with runnable examples.  
- ✅ Either a **merged module** or a **documented separation** (with rationale).  
- ✅ **Removal of `_data_old.py`** (if no deps) and notes in CHANGELOG.  
- ✅ **README** updates and cross‑links.

---

## Definition of Done (DoD)

- All KRs met; examples run in a fresh environment.  
- No references remain to `_data_old.py`.  
- CI lints/doc build pass; unit tests cover key flows (non‑streaming & streaming).  
- Docs explain **how** and **why**, not only **what**; common pitfalls called out.

---

## Risks & Potential Flaws (Mitigations)

- **Merging `store` + `data` may conflate concerns** → If merged, keep sub‑sections and a small “module map”; if separate, add clear cross‑links.  
- **Doc/code drift** → Add example‑snippets to tests or doctests where feasible.  
- **Streaming single‑consumer invariant** → Document explicitly; enforce with guard/exception in iterator if already consumed.  
- **Singleton + namespace leakage** on Bulletin → Document namespacing best‑practices and multi‑tenant safety.【8†source】  
- **Threading races** during cancel/cleanup → Document state diagram + callback ordering; add tests for edge states.

---

## Source Note

Progress and clarifications summarized from the attached Claude session log and diffs.【8†source】