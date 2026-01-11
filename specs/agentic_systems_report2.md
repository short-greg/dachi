# Agentic Systems: A Concise Report (Merged & Expanded) — 

## 1) What Is Agency?

* **Definition:** The capacity of a system to **perceive**, **decide**, and **act** toward objectives under uncertainty—controlling *when* to act, *what* to do, and *why* (via evaluation).
* **Minimal agency:** Closed-loop perception→action, even if reflexive and fixed.
* **Stronger agency:** Adds internal evaluation, prediction of consequences, autonomy of timing, and adaptation (learning).
* **Tool vs. agent:** Tool = invoked externally; Agent = initiates/chooses actions given goals and constraints.


## 2) Canonical Foundations (AIMA & Learning)

* **AIMA agent types:**

  * **Reactive (reflex):** direct percept→action.
  * **Model-based:** maintains a world/belief state.
  * **Goal-based:** plans and checks goal satisfaction.
  * **Utility-based:** trades off outcomes with a scalar objective.
  * **Learning agent:** Critic (evaluate), Learning element (update), Problem Generator (explore).

* **Learning (contrast & placement):**

  * *Samuel (1959):* “Ability to learn without being explicitly programmed.”
  * *Mitchell (1997):* Learns from experience **E** on tasks **T** measured by performance **P** if P improves with E.
  * **Clarification:** In ML, “inference” = applying a model. In logic, inference means deriving conclusions (deduction, induction, abduction). Conflating these hides the difference between *pattern application* and *true reasoning*.


## 3) Spectrum of Agent Types (Finer Levels)

**Reactive agents**

* *Level 0:* Hard-wired stimulus→response (thermostat).
* *Level 1:* Conditioned rules / small finite state machines; no temporal abstraction.

**Model-based agents**

* *Level 2a:* Shallow memory (buffers, traces).
* *Level 2b:* Predictive statistical model (next-state expectations).
* *Level 2c:* Causal/interventional model (explicit dynamics, counterfactual reasoning).

**Goal-based agents**

* *Level 3a:* Goal recognition (detect satisfied/unsatisfied states).
* *Level 3b:* Planning/search for action sequences.
* *Level 3c:* Multi-goal arbitration & constraints.

**Utility-based agents**

* *Level 4a:* Scalar utility; preference ranking over states/trajectories.
* *Level 4b:* Risk/time trade-offs (short vs. long horizon).
* *Level 4c:* Adaptive utilities (context-, user-, or learning-driven).

**Learning agents**

* *Level 5a:* Passive (improve predictions).
* *Level 5b:* Active (policy/value improvement).
* *Level 5c:* Meta-learning (learn how to learn; exploration policies).
* *Level 5d:* Continual learning in deployment (non-stationarity).


## 4) Our Agent Architecture (Decomposed & Strengthened)

**Environment ↔ Sensors / Actuators**

* *Sensors:* APIs, logs, perception pipelines (vision, text, telemetry).
* *Actuators:* System calls, tools/APIs, messaging, physical effectors.
* *Hazards:* Unreliable I/O, non-idempotent actions, partial observability.

**Percept Processing**

* Ingress → normalization → filtering → feature extraction/representation.
* *Quality gates:* schema/format checks, confidence thresholds.
* *Hazards:* Garbage-in, covariate shift, adversarial inputs.

**State Estimation / Belief (Memory-Backed)**

* Belief state = compression of history into current sufficient info.
* *Mechanisms:* Bayes filters, particle filters, learned encoders, key–value memories.
* *Memory types:* episodic (events), semantic (facts), procedural (skills/options).
* *Hazards:* Stale/contradictory memory; lack of provenance and expiration policies.

**World Model**

* *Observational:* sensor likelihoods.
* *Predictive:* next-state dynamics.
* *Causal:* counterfactual interventions.
* *Value:* assigns returns to states and actions.
* *Uncertainty:* distinguish aleatoric vs. epistemic; manage via ensembles/Bayesian methods.
* *Hazards:* Confounding, model exploitation, overconfidence.

**Goals & Constraints**

* *Representations:* goal predicates, temporal logic, hard vs. soft constraints.
* *Multi-objective:* weighted utilities, Pareto sets, aspiration levels.
* *Hazards:* Goal misspecification, reward hacking.

**Utility Function**

* *Risk sensitivity:* risk-neutral vs. risk-averse.
* *Time preference:* discounting, deadlines, value of waiting.
* *Context adaptation:* utilities that shift with user, state, or task mode.
* *Hazards:* Misaligned proxies; non-stationary preferences.

**Planner / Search**

* *Methods:* A\*, D\*, MCTS, hierarchical task networks, behavior trees.
* *Anytime behavior:* interruptible planners with bounded compute.
* *Hazards:* Myopic planning, non-recoverable branches.

**Policy / Controller (Action Selection)**

* *Reactive policies:* direct mappings.
* *Model-predictive control:* receding horizon.
* *Hybrid control:* learned policies wrapped with safety constraints.
* *Hazards:* unsafe extrapolation under distribution shift.

**Tool Interface / Execution Layer**

* *Preconditions & effects:* contracts, typed arguments.
* *Safety rails:* sandboxing, rate limits, rollback mechanisms.
* *Verification:* post-conditions, validation checks.
* *Hazards:* side effects, race conditions.

**Scheduler / Triggers (When-to-Act)**

* *Modes:* time-driven, event-driven, threshold-based, value-of-information/action.
* *Hazards:* thrashing, starvation, missed deadlines.

**Evaluation & Learning Loop (AIMA placement)**

* *Critic:* compares outcomes to standards.
* *Learning element:* updates policy/model/memory.
* *Problem generator:* proposes safe exploration.
* *Hazards:* drift, feedback loops, uncontrolled adaptation.

**Safety & Governance**

* *Self-monitoring:* uncertainty tracking, anomaly detection.
* *Human-in-the-loop:* interventions and approvals.
* *Telemetry:* audit logs, provenance, reproducibility.
* *Hazards:* silent failures, lack of accountability.

---

## 5) World Models (Why They Matter & How They Degrade)

* **Uses:** rollouts, model-predictive control, wait-vs-act decisions, counterfactuals.
* **Calibration:** reliability diagrams, ensemble spread, conformal prediction.
* **Maintenance:** shadow updates, backtesting, guarded deployment.
* **Failure modes:** policy shift, data drift, goal drift.
* **Mitigations:** continual evaluation sets, causal validation, off-policy correction.

---

## 6) Inference (Logical) vs. Association (Statistical)

* **Association:** correlation-based pattern mapping (e.g., ML runtime “inference”). Efficient but brittle.
* **Logical inference:**

  * *Deduction:* necessary conclusions.
  * *Induction:* generalization from examples.
  * *Abduction:* best explanations.
* **Operational blend:** association for recognition; inference for novel or critical decisions.
* **Design principle:** Use association for fast paths, inference for safety-critical reasoning.

---

## 7) Feedback & Rewards (External / Internal; Extrinsic / Intrinsic)

* **External feedback (grounding):** task rewards, labels, human ratings, KPIs.

  * *Extrinsic reward:* supplied by environment/supervisor (e.g., success/failure).
  * *Design tip:* use counterfactual baselines, delayed reward assignment strategies.
* **Internal feedback (autonomy):** goal satisfaction checks, constraint violations, uncertainty signals.

  * *Intrinsic reward:* curiosity, novelty, information gain, empowerment, surprise minimization.
  * *Design tip:* cap or anneal intrinsic terms to prevent distraction.
* **Blending signals:**

  * Reward shaping without altering optimal policy.
  * Multi-critic setups (task, risk, curiosity).
  * Human-in-the-loop reward adjustment.
* **Failure modes:**

  * *Reward hacking:* mitigated by constraints and audits.
  * *Myopic exploitation:* mitigated by long-horizon evaluation.
  * *Unsafe exploration:* mitigated by shielded intrinsic signals and simulators.

## 8) Intelligence Components (Revised Spectrum)

* **Minimal Core:**

  * Sense–act loop (perception + action).
  * Evaluation (did the action meet conditions?).
* **Basic Agency:**

  * Memory (episodic, semantic, procedural).
  * Simple predictive model (next-state expectations).
* **Intermediate Agency:**

  * Explicit goals and inference-based planning.
  * Ability to arbitrate between conflicting goals.
* **Advanced Agency:**

  * Utility-based decision-making.
  * Trade-offs across time, risk, and multiple objectives.
  * Causal world models for counterfactual reasoning.
* **Strong Agency:**

  * Continual learning in deployment.
  * Meta-cognition (self-monitoring, uncertainty estimates).
  * Creativity / abduction (novel hypotheses, tool creation).
  * Reflective control (knows when to override or halt itself).

---

## 9) Autonomy, Embodiment, and Decision Paradigms

* **Autonomy scales (robotics & human-factors tradition):**

  * Level 0: Manual (no autonomy).
  * Level 3–4: Shared autonomy (agent executes options, human supervises).
  * Level 6–7: Conditional autonomy (agent acts within bounded contexts).
  * Level 10: Full autonomy (self-governing, no human oversight).
  * Today’s LLM-based agents usually operate at **Level 3–5**.

* **Embodiment:**

  * *Purely cognitive agents*: outputs = text/predictions.
  * *Digital agents*: APIs, databases, software actuators.
  * *Physical agents*: robots, drones, IoT devices.
  * Embodiment defines the scope and **stakes** of agency.

* **Decision paradigms:**

  * *Symbolic/logical inference* (rules, ontologies, theorem proving).
  * *Search/planning* (A\*, HTN, MCTS).
  * *Learning-based control* (RL, policy gradient).
  * *Heuristics/metaheuristics* (satisficing, swarm, evolutionary).
  * Note: Agency ≠ decision method. Agency is *having a loop*. Method is *how the loop is filled*.

---

## 10) Bounded Rationality and Uncertainty

* **Bounded rationality (Herbert Simon):** Agents cannot compute or sense everything, so they *satisfice*—settle for “good enough.”
* **Implication for design:** Utility and goal systems must tolerate approximate reasoning, incomplete information, and limited planning depth.
* **Uncertainty categories:**

  * *Aleatoric:* stochastic environment randomness.
  * *Epistemic:* ignorance or missing knowledge.
* **Approaches:**

  * Probabilistic inference, Bayesian updating.
  * Risk-sensitive planning (CVaR, variance-aware objectives).
  * Confidence calibration, ensembles, conformal prediction.

---

## 11) Multi-Agent Systems

* **Individual vs. multi-agent:** A single agent optimizes only for itself; multi-agent systems must coordinate.
* **Modes of interaction:**

  * *Cooperative:* joint planning, resource sharing.
  * *Competitive:* adversarial optimization, game theory.
  * *Mixed-motive:* partial cooperation with conflicting incentives.
* **Capabilities needed:**

  * Negotiation, trust modeling, norm-following.
  * Emergent communication protocols.
* **Risks:** Coordination failures, deception, collusion.

---

## 12) Ethics, Safety, and Governance

* **Safety scaffolds:**

  * Guardrails, sandboxing, constraint satisfaction layers.
  * Fallback and “safe stop” states.
* **Governance:**

  * Audit logging, provenance, human approval checkpoints.
  * Role-based access to actuators.
* **Alignment approaches:**

  * Minimal: hard-coded constraints.
  * Intermediate: preference learning, corrigibility.
  * Advanced: constitutional or norm-based alignment.
* **Ethical implication:** Agency implies accountability. The more autonomy, the more critical governance becomes.

---

## 13) Emergent Properties at Higher Levels

* **Meta-cognition:** The agent monitors its own reasoning quality, detects contradictions, and signals uncertainty.
* **Creativity / abduction:** Generates hypotheses or new strategies not explicitly trained for.
* **Self-reflection:** Evaluates not just external outcomes but its *own processes* (“Am I reasoning well? Should I stop?”).
* **Exploration:** Problem Generator proposes new tasks or directions (curiosity-driven learning).

---

## 14) Practical Build Path (Progressive)

1. Reflex tool (hard-coded responses).
2. Percept processing & feature extraction.
3. Memory (episodic/semantic).
4. Simple world model (associative → statistical → causal).
5. Goal representation & goal satisfaction checks.
6. Utility/trade-offs for prioritization.
7. Evaluation loops (external + internal feedback).
8. Learning (safe, gradual, monitored).
9. Scheduler/when-to-act triggers.
10. Planner/search integration.
11. Critic/Learning/Exploration loop (AIMA full architecture).
12. Safety, governance, and alignment.

**Stop rule:** Add only what addresses observed deficiencies. E.g.:

* Coverage gaps → world model.
* Conflicting goals → utility.
* Untimely responses → scheduler.
* Drift → learning.
* Safety risk → guardrails.

---

## 15) Bottom Line

* **Agency is graded, not binary.** Reflexes, models, goals, utilities, learning—each adds depth.
* **Critical transitions:**

  * Reflex → model (prediction).
  * Goal → utility (trade-offs).
  * Utility → learning (adaptation).
* **Inference + association:** Association delivers efficiency; inference delivers flexibility. Both are necessary.
* **Feedback:** Extrinsic/external signals ground the agent; intrinsic/internal signals enable autonomy. Balance prevents both passivity and reward hacking.
* **Design reality:** Today’s “agents” are mostly orchestration shells. To reach *true agency*, we need autonomy triggers, explicit evaluation, world models, utility functions, and feedback loops—all under bounded rationality and governance.

