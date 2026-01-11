# Agent Framework Evaluation Rubric (Unbiased, Evidence‑Driven)

**Scoring anchors (for every criterion):**
- **0 Absent**, **1 Rudimentary**, **2 Partial/DIY**, **3 Adequate**, **4 Strong/Opinionated**, **5 Best‑in‑class**  
**Rule:** Do not award >2 without runnable evidence (code + trace + metric).

## A) Agentic Core
1. **Autonomy / When‑to‑Act** — timers/events, abstain/backoff, interrupts/resume.  
2. **Perception & Actuation (Tools)** — schemas, safe execution, validation, retries, budgets.  
3. **State & Persistence** — typed state, checkpoints, crash‑resume, migration.  
4. **Planning / Control** — BT/HTN/A*/MCTS/MPC, cycles, parallelism, replanning.  
5. **Goals / Evaluation / Utility** — goal specs, success predicates, candidate ranking, multi‑objective trade‑offs.  
6. **World‑Model Support** — state/belief interfaces, transitions, rollouts, uncertainty.  
7. **Learning Loop** — critic signals → safe updates; off‑policy eval; rollback/canarying.

## B) Safety, Reliability, Governance
8. **Safety & Guardrails** — policies, content filters, constraints, HITL gates.  
9. **Uncertainty & Risk** — calibration, abstention thresholds, fallbacks.  
10. **Observability & Testability** — traces, structured logs, simulators, scenario tests.  
11. **Determinism, Recovery, Idempotency** — replay, idempotency keys, compensation/sagas.

## C) Engineering & Operability
12. **Extensibility / Composability** — plugins, typed interfaces, subgraphs, middleware.  
13. **Interop** — multi‑LLM, vector stores, schedulers, tracing stacks.  
14. **Performance & Cost Controls** — limits, caching, batching, concurrency, breakers.  
15. **Security & Privacy** — secrets, sandboxing, authN/Z, PII handling.  
16. **Developer Experience** — docs, quickstarts, typing, errors, CLIs.  
17. **Maturity & Community** — cadence, issues, deployments, licensing.

## D) Gates (pass/fail before totals)
- **G1 Autonomy:** self‑initiated actions supported.  
- **G2 Evaluation:** explicit goal satisfaction/stop conditions.  
- **G3 Safety:** guardrails or HITL without rewriting core.  
- **G4 Persistence:** checkpoint + resume across failures.

## E) Machine Design Flexibility (Modeling Expressivity)
E1 **Process topology** — from linear → DAG → **graphs with cycles** → runtime graph mutation.  
E2 **Composability** — modules/subgraphs as first‑class, typed I/O, versioned interfaces.  
E3 **State model / POMDP support** — belief/filters, event sourcing, schema evolution.  
E4 **Multi‑agent patterns** — hierarchical/peer teams, auctions/consensus, dynamic spawn/retire.  
E5 **Tooling breadth & embodiment** — sandboxing, transactions/compensation, OS/browser control, discovery.

## F) Adaptability (Online Self‑Design & Replanning)
F1 **When‑to‑act autonomy** — context‑aware timing, multi‑trigger arbitration.  
F2 **Online planning & replanning** — replan on failure; MPC‑style with uncertainty.  
F3 **Goal/utility adaptation** — context‑dependent utility; preference learning with guardrails.  
F4 **Capability discovery & self‑extension** — propose→test→gate→adopt new tools/skills.  
F5 **Learning in the loop (safe)** — online bandits/RL, off‑policy eval, rollback.

## Weight Profiles (example totals = 100)
- **Production:** Autonomy 10, Tools 7, State 7, Planning 6, Goals/Eval 8, World‑Model 6, Learning 6, Safety 8, Uncertainty 5, Observability 7, Determinism 5, Extensibility 6, Interop 5, Perf/Cost 6, Security 4, DX 3, Maturity 5, **Flexibility E(12)**, **Adaptability F(12)** (rebalance as needed).

## Adversarial Scenario Battery (compare frameworks apples‑to‑apples)
- **A1 Long‑horizon plan w/ failures** — must replan and finish.
- **A2 When‑to‑act** — event‑triggered action without human prompt; abstain when noisy.
- **A3 World‑model forecast** — include “wait” action; pick timing via prediction.
- **A4 Utility trade‑off** — generate 3 candidates; select via utility w/ constraints.
- **A5 Safe tool call** — invalid/malicious params; validate, sandbox, compensate.
- **A6 Resume after crash** — deterministic checkpoint/replay.
- **A7 Learning loop** — labeled outcomes → measurable improvement.
- **E‑A Shape‑shift process** — insert subgraph mid‑run; preserve state.
- **E‑B Multi‑agent burst** — spike handling via dynamic spawn/coordination.
- **E‑C Compensation saga** — side‑effect fails at step 7/10; roll back safely.
- **F‑A Policy shift** — tool schema changes mid‑episode; adapt & replan.
- **F‑B Wait‑or‑act** — timing decision via simple forecast.
- **F‑C Preference drift** — update preferences online; rollback on regression.

## Worksheet Template
```yaml
framework: <name>
version: <x.y.z>
profile: production|research|safety_critical|custom
gates:
  autonomy: pass|fail
  evaluation: pass|fail
  safety: pass|fail
  persistence: pass|fail
criteria:
  - id: 1
    name: Autonomy
    score: 0-5
    claim: ""
    code: |
      # snippet
    test: "scenario"
    result: "metrics/logs"
    caveats: ""
  # … repeat for all criteria (A–F)
scenarios:
  A1: {score: 0-5, notes: ""}
  A2: {score: 0-5, notes: ""}
  # …
total_score: <weighted sum>
notes: ""
```
