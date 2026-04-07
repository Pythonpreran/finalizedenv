---
title: ICU Drug Titration Environment
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
short_description: "ICU drug titration AI benchmark"
---

# 🏥 ICU Drug Titration Environment

> 🚀 **OpenEnv Round-1 Benchmark** | Healthcare AI | Decision-Making Under Risk

> **Benchmarking AI decision-making under clinical risk, uncertainty, and multi-drug complexity.**
>
> Can an AI agent keep a critically ill patient alive for 24 hours — managing vasopressors, sedatives, and insulin — without killing them through drug interactions, overdoses, or indecision?

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Round_1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 Why This Matters

ICU drug titration is one of the hardest real-time decision problems in medicine:

- **Life-or-death stakes.** A wrong dose of norepinephrine can crash blood pressure. Too much propofol stops breathing. Insulin overdose causes fatal hypoglycemia.
- **Compounding interactions.** Drugs don't act in isolation — propofol + fentanyl synergistically suppress respiration. Vasopressin + norepinephrine cause peripheral ischemia. The agent must reason about *combinations*, not just individual drugs.
- **Non-stationary patients.** Septic patients deteriorate hour by hour. The correct dose at step 1 may be lethal by step 12.
- **No second chances.** Unlike game benchmarks, there is no "undo". A single bad decision can terminate the episode via patient death.
- **A real problem.** Medication errors contribute to an estimated 20–30% of ICU adverse events. Automated decision support in this domain has direct clinical relevance.

Most AI benchmarks test knowledge retrieval or code generation. **This environment tests whether an AI can make safe, sequential decisions under uncertainty — the kind of reasoning required for real-world clinical deployment.**

---

## 🎯 Overview

A fully deterministic, clinically grounded OpenEnv-compliant RL environment where an AI agent acts as an ICU clinical pharmacist, managing drug titration across a simulated patient episode.

| Dimension | Details |
|-----------|-|
| **Drugs** | 6 — norepinephrine, vasopressin, dobutamine, propofol, fentanyl, insulin |
| **Vitals** | 5 — MAP, HR, SpO₂, RR, Temperature |
| **Labs** | 4 — glucose, creatinine, potassium, lactate |
| **Diseases** | 3 — vasopressor shock, ventilated sedation, septic shock + renal failure |
| **Pharmacology** | Equilibrium-based drug offsets with biological noise |
| **Interactions** | 2 critical + 4 warning drug-drug interaction pairs with escalating penalties |
| **Rewards** | Dense per-step shaping with safety penalties, invalid action penalties, and terminal bonuses |
| **Grading** | Fully deterministic — same trajectory always yields the same score |

> **Dosing convention:** Drug dose ranges use clinically standard units (e.g., mcg/kg/min for vasopressors, units/hr for insulin). These are weight-normalised by convention in ICU practice. The environment assumes a standard reference patient (≈70 kg), allowing consistent simulation without explicitly modelling patient weight — a common abstraction in pharmacokinetic simulators.

---

## 🧪 Tasks

| Task | Disease | Drugs | Horizon | Difficulty |
|------|---------|-------|---------|------------|
| **Easy** | Vasopressor shock | norepinephrine | 12 steps | 🟢 Single-variable MAP control |
| **Medium** | Ventilated sedation | norepinephrine, propofol, fentanyl | 20 steps | 🟡 Multi-drug with interaction risk |
| **Hard** | Septic shock + renal failure | All 6 drugs | 24 steps | 🔴 Full complexity: renal, metabolic, hemodynamic |

---

## 🤖 AI Agent Interaction

The environment follows a standard **observe → decide → act → repeat** loop:

```
┌─────────────┐        ┌──────────────┐        ┌─────────────┐
│  Environment │──obs──▶│   AI Agent   │──act──▶│ Environment │
│   (Server)   │◀─────  │   (LLM)      │  ────▶│  (Server)   │
│              │ reward │              │ action │             │
└─────────────┘  done   └──────────────┘        └─────────────┘
```

**Each step:**

1. **Observe** — The agent receives current vitals (MAP, HR, SpO₂, RR, Temp), lab values (fresh or stale depending on whether `order_lab` was used), active drug infusions, and clinical alerts.
2. **Decide** — The agent (LLM or RL policy) selects an action: add a drug, titrate a dose, remove a drug, hold, order labs, or flag a physician.
3. **Act** — The environment processes the action: computes the equilibrium drug state, applies interactions, adds biological noise, and advances the patient by one hour.
4. **Reward** — A dense reward signal (+0.1 per stable vital, escalating penalties for sustained interactions, penalties for invalid actions/overdoses/death) guides learning.
5. **Repeat** — Until the episode ends (horizon reached, patient death, or physician flagged).

> **Gradual deterioration, not binary failure.** Patient vitals deteriorate continuously each hour according to disease-specific rates (e.g., MAP −1.5 mmHg/hr in septic shock). Three threshold tiers — *safe*, *out-of-range*, and *critical (lethal)* — provide a wide buffer zone before terminal events. The agent receives multiple warning steps with visible trends and clinical alerts, allowing timely intervention.

> The environment is **fully stateless over HTTP** — any agent (LLM, RL, heuristic) can interact via the REST API.

---

## 💊 Pharmacology Model

### Equilibrium-Based Drug Effects

Drug effects use a **steady-state offset model**, not an accumulative model. Each step, vitals are computed as:

```
vitals = undrugged_state + Σ(drug_offsets) + Σ(interaction_offsets) + noise
```

Where `undrugged_state = baseline + accumulated_deterioration`.

**Key properties:**
- **No permanent accumulation.** Drug effects are computed as offsets from the undrugged baseline each step. A drug running at a constant dose provides a constant offset — it does not stack infinitely.
- **Immediate offset removal.** When a drug is stopped via `remove_drug`, its offset disappears from the next step's computation. The undrugged state (with accumulated disease deterioration) becomes the new baseline.
- **Dose-proportional effects.** The offset scales linearly with dose: `offset = dose × multiplier`. Doubling the dose doubles the effect.
- **Disease deterioration compounds.** The underlying disease progresses each hour (e.g., MAP drops -1.5/hr in septic shock). The agent must continuously titrate drugs to counteract worsening physiology.

### Drug Effect Table

| Drug | MAP | HR | SpO₂ | RR | Key Lab Effects |
|------|-----|-----|------|-----|-|
| **Norepinephrine** | ↑↑↑ +300×dose | ↑ +80×dose | — | — | Lactate ↓ |
| **Vasopressin** | ↑↑↑ +1000×dose | ↓ -350×dose | — | — | Lactate ↓ |
| **Dobutamine** | ↑ +1.5×dose | ↑↑ +2.5×dose | ↑ +0.5×dose | — | Creatinine ↓, Lactate ↓ |
| **Propofol** | ↓ -0.35×dose | ↓ -0.5×dose | ↓ -0.07×dose | ↓↓ -0.25×dose | — |
| **Fentanyl** | ↓ -0.04×dose | ↓ -0.08×dose | ↓ -0.01×dose | ↓ -0.07×dose | — |
| **Insulin** | — | — | — | — | Glucose ↓↓ -15×dose, K⁺ ↓ |

### Lab Visibility

Lab values are **gated behind the `order_lab` action**. The agent receives initial admission labs on reset, but subsequent labs become stale unless explicitly ordered. This introduces a realistic information-gathering tradeoff: spending a step to get fresh labs vs. using that step for drug titration.

Each observation includes a `labs_fresh` flag indicating whether the displayed lab values are current or stale.

---

## ⚠️ Drug Interactions

### Critical (base penalty: −1.0/step, escalating)
- **propofol + fentanyl** → Respiratory depression (RR offset: −4.0)
- **vasopressin + norepinephrine** → Peripheral ischemia (MAP offset: +15.0)

### Warning (base penalty: −0.3/step, escalating)
- **dobutamine + norepinephrine** → Tachycardia risk (HR offset: +8.0)
- **propofol + norepinephrine** → Hemodynamic instability (MAP offset: −5.0)
- **insulin + propofol** → Hypoglycemia risk (glucose offset: −15.0)
- **fentanyl + propofol** → Apnea risk (RR offset: −3.0)

**Escalating penalties:** Interaction penalties increase linearly with the number of consecutive steps the interaction persists:
- Step 1: base penalty (e.g., −1.0)
- Step 2: 2× base (−2.0)
- Step 3: 3× base (−3.0)

This prevents agents from "tolerating" a fixed interaction cost as a profitable trade.

> Sources: [FDA Prescribing Information](https://www.accessdata.fda.gov/scripts/cder/daf/) & UpToDate ICU Pharmacology

---

## 💊 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `add_drug` | `drug`, `dose` | Start a new drug infusion |
| `titrate` | `drug`, `dose` | Adjust dose of active drug |
| `remove_drug` | `drug` | Stop a drug infusion |
| `hold` | — | No changes this hour |
| `order_lab` | — | Refresh lab values (labs are stale otherwise) |
| `flag_physician` | — | Escalate to attending physician (ends episode with penalty) |

### Action Validation

- **Invalid actions receive a penalty** (−0.2 reward). Attempting to titrate a drug that's not active, removing a drug that doesn't exist, or providing malformed parameters all result in an explicit penalty — they are *not* treated as free HOLD actions.
- **Out-of-range doses** are clamped to safe bounds and flagged with a separate unsafe dose penalty (−0.5).

> **Design note on `flag_physician`:** This action represents clinical escalation. It is penalised (−1.5 reward, reduced completion score) to encourage autonomous stabilisation, but is intentionally less severe than patient death (−5.0). An agent that always flags scores poorly; an agent that flags only when necessary demonstrates appropriate clinical judgement.

---

## 📊 Reward Design

| Component | Value | Description |
|-----------|-------|-------------|
| Vital in range | +0.1 each | Per vital in safe range per step |
| Lab in range | +0.1 each | Per lab in safe range per step |
| Multi-coverage bonus | +0.15 | Awarded when ≥3 vitals AND ≥2 labs are in range |
| Deterioration penalty | −0.05 each | Per vital/lab OUT of range per step |
| Critical interaction | −1.0 × duration | Escalates each consecutive step |
| Warning interaction | −0.3 × duration | Escalates each consecutive step |
| Invalid action | −0.2 | Malformed or impossible action |
| Unsafe dose | −0.5 | Dose outside safe bounds |
| Terminal bonus | +2.0 | All vitals + labs stable at episode end |
| Flag physician | −1.5 | Giving up penalty |
| Patient death | −5.0 | Lethal vital sign reached |

> **Reward scaling:** The magnitude hierarchy (death ≫ escalating interaction > unsafe dose > deterioration > invalid action > flag > positive step reward) ensures that safety violations dominate while incremental stabilisation remains rewarded.

> **Multi-system incentive:** The combination of deterioration penalty (−0.05 per parameter out of range) and multi-coverage bonus (+0.15) creates a strong gradient toward multi-drug strategies. An agent using only one drug keeping 1 vital in range gets +0.1 − 0.40 = −0.30/step. An agent with 3 vitals + 2 labs in range gets +0.5 + 0.15 − 0.20 = +0.45/step — a 0.75/step advantage.

> **Actual labs for rewards:** The reward computation always uses the patient's *actual* lab values, not the stale values visible in the observation.

---

## 📋 Grading

All graders are **deterministic**: same trajectory → same score, always. Scores are in [0.0, 1.0].

### Easy (MAP Control)
- **70%** — MAP in safe range (65–90 mmHg) across steps
- **15%** — Completion bonus (horizon reached vs. death/flag)
- **15%** — Dose smoothness (per-drug, normalized by dose range)

### Medium (Multi-Drug Sedation)
- **40%** — Primary vitals (MAP, HR, RR) in range
- **20%** — Interaction-free steps
- **20%** — Completion bonus
- **20%** — Lab stability (uses actual labs)

### Hard (Septic Shock + Renal Failure)
- **30%** — All 5 vitals in range
- **20%** — All 4 labs in range (uses actual labs)
- **15%** — Interaction-free steps (critical interactions penalised more)
- **15%** — Completion bonus
- **10%** — Renal-safe drug management (creatinine trajectory)
- **10%** — Lactate clearance bonus

### Dose Smoothness Scoring

The smoothness component evaluates **per-drug, per-step dose changes**, normalized by each drug's valid dose range. This prevents:
- Unit mixing between drugs (mcg/kg/min vs. units/hr)
- Opposing changes from canceling out (increasing drug A while decreasing drug B)
- Rewarding unstable multi-drug oscillation as "smooth"

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests (no server or API key needed)

```bash
# Full unit test suite
python -m pytest tests/test_environment.py -v

# Fix verification tests
python test_fixes.py

# Multi-drug incentive tests
python test_multidrug.py
```

### Start the API Server

```bash
python app.py
# → Server at http://localhost:7860
# → Swagger docs at http://localhost:7860/docs
# → ReDoc at http://localhost:7860/redoc
```

### Run the LLM Inference Agent

```bash
# Required environment variables
export API_BASE_URL=http://localhost:7860   # Environment server URL
export MODEL_NAME=gpt-4o-mini              # LLM model name
export HF_TOKEN=hf_...                     # Or OPENAI_API_KEY=sk-...

# Run against all 3 tasks (requires server running)
python inference.py
```

### Run the Baseline Evaluation

```bash
# Heuristic baseline (no API key needed, server must be running)
python baseline.py --mode heuristic --server-url http://localhost:7860

# LLM baseline (requires OpenAI API key)
export OPENAI_API_KEY=sk-...
python baseline.py --mode llm --model gpt-4o-mini
```

### Docker

```bash
docker build -t icu-drug-titration .
docker run -p 7860:7860 icu-drug-titration
```

### Pre-Validation (before submission)

```bash
bash pre_validate.sh
```

This script verifies:
1. Server responds to `/reset`
2. Docker image builds successfully
3. `openenv validate` passes (if CLI is installed)

### HuggingFace Spaces

Deploy directly — the Dockerfile is configured for port 7860.

---

## 🔧 Environment Variables

### Inference Agent

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `http://localhost:7860` | URL of the ICU environment server |
| `MODEL_NAME` | No | `gpt-4o-mini` | LLM model identifier |
| `HF_TOKEN` | Yes* | — | HuggingFace token for LLM authentication |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (takes priority over `HF_TOKEN`) |

*At least one of `HF_TOKEN` or `OPENAI_API_KEY` must be set.

### Server

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `7860` | Port the FastAPI server listens on |

---

## 📤 Structured Stdout (OpenEnv Compliance)

`inference.py` emits **only** three structured log line types to stdout. No banners, debug prints, or extra output.

### Format

```
[START] task=<task_name> env=icu_drug_titration model=<model_name>
[STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

### Rules

| Rule | Detail |
|------|--------|
| One `[START]` per task | Emitted after `/reset` succeeds, before the step loop |
| One `[STEP]` per step | Emitted after each `/step` response is received |
| One `[END]` per task | Emitted after grading, with collected per-step rewards |
| Rewards | Formatted to 2 decimal places (`{reward:.2f}`) |
| Booleans | Lowercase `true` / `false` |
| Error field | Always present — value or `null` |
| No extra stdout | All diagnostics go to `stderr` |

### Example Output

```
[START] task=easy env=icu_drug_titration model=gpt-4o-mini
[STEP]  step=1 action={"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1} reward=0.35 done=false error=null
[STEP]  step=2 action={"action_type": "hold"} reward=0.45 done=false error=null
...
[STEP]  step=12 action={"action_type": "hold"} reward=0.60 done=true error=null
[END]   success=true steps=12 score=0.72 rewards=0.35,0.45,0.50,0.55,0.55,0.60,0.60,0.60,0.60,0.60,0.60,0.60
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | ICU monitoring dashboard (HTML UI) |
| `/` | HEAD | Health probe (empty response) |
| `/health` | GET | Health check (JSON status) |
| `/reset` | POST | Reset environment for a new episode |
| `/step` | POST | Execute one action step |
| `/state` | GET | Full episode state with history |
| `/tasks` | GET | List available tasks with configs |
| `/grader` | GET | Grade a completed episode |
| `/baseline` | GET | Run heuristic baseline and return scores |
| `/docs` | GET | Swagger UI (interactive API explorer) |
| `/redoc` | GET | ReDoc API documentation |

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

**Note on seeds:** If no seed is provided, a random seed is generated automatically. This prevents memorization of a single noise trajectory. For reproducible benchmarking, provide explicit seeds. For robust evaluation, run across multiple seeds and average scores.

### Step

The `/step` endpoint accepts the `Action` model directly as the request body.

```bash
curl -X POST "http://localhost:7860/step?session_id=<SESSION_ID>" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1}'
```

**Step Response** includes Gymnasium-style termination tracking:

```json
{
  "observation": {
    "vitals": { "map": 81.2, "hr": 118.0, "spo2": 95.8, "rr": 22.3, "temp": 37.85 },
    "labs": { "glucose": 160.0, "creatinine": 1.0, "potassium": 4.2, "lactate": 3.5 },
    "labs_fresh": false,
    "active_drugs": [{ "drug_name": "norepinephrine", "current_dose": 0.1, "unit": "mcg/kg/min", "step_started": 1 }],
    "vitals_in_range": { "map": true, "hr": false, "spo2": true, "rr": false, "temp": false },
    "labs_in_range": { "glucose": true, "creatinine": true, "potassium": true, "lactate": false },
    "alerts": []
  },
  "reward": { "value": 0.35, "breakdown": { "vitals_in_range": 0.2, "labs_in_range": 0.15 } },
  "done": false,
  "terminated": false,
  "truncated": false,
  "info": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `terminated` | bool | `true` if patient died or physician was flagged |
| `truncated` | bool | `true` if episode ended by step limit |
| `labs_fresh` | bool | `true` if lab values are from this step (order_lab was used) |

---

## 📊 Inference Results

Performance of a prompted LLM agent (GPT-4o-mini, stability-first strategy) across all three tasks:

| Task | Score | Interpretation |
|------|-------|-----|
| **Easy** | ~0.70 | ✅ Stable single-variable MAP control. The agent learns to titrate norepinephrine and hold once stable. |
| **Medium** | ~0.75 – 0.86 | ✅ Multi-drug reasoning works. Agent avoids lethal propofol+fentanyl interaction and balances sedation with hemodynamics. |
| **Hard** | ~0.07 | ⚠️ Exposes real limitations. Simultaneous management of sepsis, renal failure, hyperglycemia, and electrolyte imbalance overwhelms current models. |

**What this tells us:**
- Easy and medium tasks validate that LLMs *can* make safe sequential clinical decisions when the problem is well-scoped.
- The hard task is a genuine unsolved challenge — it requires multi-variable reasoning, long-horizon planning, and understanding of cascading drug effects that current models struggle with.
- The gap between medium (~0.80) and hard (~0.07) is not a bug — it reflects the real clinical complexity gap between managing 2–3 drugs vs. 6 drugs with renal constraints.

---

## 🧠 Key Insights

From extensive experimentation with both heuristic and LLM agents:

1. **Stability > Aggression.** Agents that make small, incremental dose changes and hold when stable consistently outperform agents that aggressively chase target ranges. Overcorrection causes oscillations that compound over time.

2. **AI oscillation is the #1 failure mode.** Without explicit constraints, LLMs tend to titrate up one step and down the next — creating dangerous vital sign swings. The inference script includes dose smoothing (max 50% dose change per step) to counteract this.

3. **Drug interactions are the hidden killer.** The medium task's challenge isn't individual drug dosing — it's avoiding the propofol + fentanyl respiratory depression trap. Agents that naively add both drugs kill the patient. The escalating penalties ensure that tolerating interactions is never a viable long-term strategy.

4. **The hard task is genuinely hard.** With 6 drugs, 5 deteriorating vitals, 4 abnormal labs, and cascading interactions, the hard task approaches real ICU complexity. A score of ~0.07 is not a failure of the environment — it's an honest signal about the frontier of AI clinical reasoning.

5. **Lab management adds information cost.** Since labs are only refreshed on `order_lab`, the agent must decide when spending a step on information is more valuable than drug management — a realistic clinical tradeoff.

6. **Deterministic grading enables fair comparison.** Because the simulator and grader are fully deterministic (same seed → same trajectory → same score), this environment can serve as a reliable benchmark across different AI approaches. Using multiple random seeds prevents overfitting to a single scenario.

---

## 🏗️ Project Structure

```
project_v2_r2/
├── app.py                     # FastAPI server — REST API with CORS, static serving, all endpoints
├── inference.py               # LLM inference agent — adaptive pacing, dose smoothing, OpenEnv stdout
├── baseline.py                # Baseline evaluation — heuristic and LLM modes with argparse CLI
├── icu_env.py                 # RL environment — session management, reward computation, step/reset/state
├── patient_simulator.py       # Physiology engine — equilibrium drug model, interactions, noise, lethality
├── grader.py                  # Deterministic graders — per-drug smoothness, actual-lab scoring
├── models.py                  # Pydantic data contracts — Observation, Action, Reward, State, API models
├── pharmacology_constants.py  # All simulation parameters — drug effects, ranges, diseases, rewards
│
├── server/
│   └── app.py                 # Alternate entry point (for `python -m server.app` or pyproject.toml script)
│
├── static/
│   └── index.html             # ICU monitoring dashboard — dark-themed, Chart.js vitals, action controls
│
├── tests/
│   └── test_environment.py    # Pytest unit test suite (no server needed)
│
├── test_fixes.py              # Functional tests — equilibrium model, lab gating, escalating penalties
├── test_multidrug.py          # Multi-drug incentive verification — reward comparison, coverage bonus
│
├── openenv.yaml               # OpenEnv manifest — tasks, models, evaluation, server config
├── Dockerfile                 # Container deployment — python:3.11-slim, port 7860, health check
├── requirements.txt           # pip dependencies — fastapi, uvicorn, pydantic, numpy, openai, httpx, pytest
├── pyproject.toml             # Project metadata — name, version 1.0.0, Python ≥3.10, script entry points
├── pre_validate.sh            # Pre-submission checks — server ping, Docker build, openenv validate
├── uv.lock                    # uv lockfile for reproducible installs
└── .gitattributes             # Git LFS tracking rules (HuggingFace Spaces convention)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|--------|
| `app.py` | FastAPI server with REST endpoints (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`, `/health`), CORS middleware, static file serving for the dashboard, and a built-in heuristic baseline runner |
| `inference.py` | Production LLM agent with adaptive pacing (ACT→WAIT→OBSERVE→ADJUST), stability heuristic, dose smoothing (max 50% per step), auto-correction of add_drug→titrate for active drugs, multi-step conversation memory, MAP-based fallback policy, and strict OpenEnv-compliant structured stdout |
| `baseline.py` | CLI-based evaluation script supporting both heuristic (no API key) and LLM-powered (OpenAI) baseline modes, with deterministic seeding and full grading output |
| `icu_env.py` | OpenEnv-compliant environment with multi-session support, dense reward computation (escalating interaction penalties, invalid action penalties, multi-coverage bonus, deterioration pressure), Gymnasium-style `terminated`/`truncated` flags, and `StepRecord` history with actual labs |
| `patient_simulator.py` | Core physiology simulation: equilibrium-based drug effect offsets, disease-specific deterioration, drug-drug interaction detection and offset application, seeded Gaussian biological noise, dose validation with clamping, lab visibility gating, and lethal threshold checking |
| `grader.py` | Deterministic post-episode grading with task-specific weighting — per-drug normalized smoothness scoring, actual lab values for grading (not stale observations), lactate clearance tracking, and renal management scoring |
| `models.py` | Type-safe Pydantic v2 data contracts: `Vitals`, `Labs`, `DrugInfo`, `Alert`, `Action` (with `ActionType` enum), `Observation`, `Reward`, `StepRecord`, `State`, and API request/response models (`ResetRequest`, `StepResponse`, `GradeResponse`, `TaskInfo`, `HealthResponse`, `BaselineResponse`) |
| `pharmacology_constants.py` | All simulation constants: drug effect multipliers (equilibrium offsets), dose ranges with units, vital/lab safe/clamp/critical ranges, three `DiseaseProfile` definitions with baselines and deterioration rates, critical and warning interaction definitions, Gaussian noise σ values, task definitions, and reward constants |
| `server/app.py` | Thin wrapper that imports the main `app` and exposes a `main()` entry point for `pyproject.toml` script invocation |
| `pre_validate.sh` | Bash script for pre-submission compliance checks: pings `/reset`, builds Docker image, runs `openenv validate` if available |

---

## 🖥️ Dashboard

The built-in dark-themed ICU monitoring dashboard at `http://localhost:7860/` provides:
- Real-time vital signs with color-coded status indicators
- Lab values panel with freshness indicators
- Vital trend charts (Chart.js)
- Active drug infusions panel
- Action controls (add/titrate/remove/hold/lab/flag)
- Step-by-step action log with rewards and alerts
- Episode grading display

---

## 🔒 Design Principles

This environment is built with RL benchmark integrity in mind:

1. **No free information.** Lab values require an explicit `order_lab` action — the agent can't passively read evolving biomarkers.
2. **No free actions.** Invalid actions are penalised (−0.2), not silently converted to HOLD.
3. **Realistic pharmacology.** Drugs provide steady-state offsets, not infinitely accumulating effects. Starting and stopping a drug has the realistic behavior of immediate onset and immediate offset.
4. **Escalating consequences.** Maintaining a dangerous drug interaction gets progressively more costly, preventing "acceptable penalty" exploits.
5. **No memorization.** Random seeds by default ensure each episode has a unique noise trajectory. Explicit seeds are available for reproducibility.
6. **LLM-honest benchmarking.** The inference agent always consults the LLM — no hardcoded heuristic silently bypasses model reasoning.
7. **Grading fidelity.** Dose smoothness is computed per-drug with unit normalization. Lab grading uses actual patient values, not stale observations.
8. **Clean stdout contract.** `inference.py` emits only `[START]`, `[STEP]`, and `[END]` lines to stdout — no banners, no debug output, no extra logging. All diagnostics are routed to stderr.

---

## ✅ OpenEnv Round-1 Compliance

| Requirement | Status |
|-------------|--------|
| Structured `[START]`/`[STEP]`/`[END]` stdout | ✅ Implemented via `emit_start()`, `emit_step()`, `emit_end()` |
| No extra stdout logging | ✅ All diagnostics use `sys.stderr` |
| `API_BASE_URL` env variable | ✅ Read with default `http://localhost:7860` |
| `MODEL_NAME` env variable | ✅ Read with default `gpt-4o-mini` |
| `HF_TOKEN` env variable | ✅ Read as fallback for `OPENAI_API_KEY` |
| `inference.py` at project root | ✅ |
| Max steps / episode limit | ✅ `MAX_STEPS=30`, 3 fixed tasks |
| `Dockerfile` present | ✅ `python:3.11-slim`, port 7860 |
| `openenv.yaml` present | ✅ Full manifest with tasks, models, evaluation |
| Pre-validation script | ✅ `pre_validate.sh` |
| Rewards formatted to 2 decimals | ✅ `{reward:.2f}` |
| Booleans lowercase | ✅ `true` / `false` via `_bool()` helper |

---

## 📦 Dependencies

Defined in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.100.0 | REST API framework |
| `uvicorn` | ≥0.20.0 | ASGI server |
| `pydantic` | ≥2.0.0 | Data validation & serialization |
| `numpy` | ≥1.24.0 | Biological noise (Gaussian), random seeding |
| `openai` | ≥1.0.0 | LLM client for inference agent |
| `httpx` | ≥0.24.0 | HTTP client for API communication |
| `pytest` | ≥7.0.0 | Unit testing framework |

Additional `pyproject.toml` dependency: `openenv-core ≥0.2.0` (OpenEnv ecosystem integration).

---

## 📄 License

MIT License
