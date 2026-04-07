"""
Inference Script for ICU Drug Titration Environment.

Runs a real LLM agent (OpenAI client) against the ICU Drug Titration
OpenEnv environment. The agent acts as an expert ICU clinical pharmacist,
making drug titration decisions step-by-step to stabilise patient vitals.

Features:
    - Modular architecture: run_task, call_llm, parse_action, fallback_action
    - Robust JSON parsing with code-fence stripping
    - MAP-based fallback policy guaranteeing stable execution
    - Detailed clinical system & user prompts
    - Clean step-by-step logging with final grading

Usage:
    # Set required environment variables
    export OPENAI_API_KEY=sk-...        # or HF_TOKEN
    export MODEL_NAME=gpt-4o-mini       # optional, defaults to gpt-4o-mini
    export API_BASE_URL=http://...      # optional, defaults to http://localhost:7860

    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write("ERROR: openai package is required. Install with: pip install openai\n")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Resolve API key: prefer OPENAI_API_KEY, fall back to HF_TOKEN
_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
if not _api_key:
    sys.stderr.write("ERROR: Set OPENAI_API_KEY or HF_TOKEN environment variable.\n")
    sys.exit(1)

MAX_STEPS: int = 30
TASKS: List[str] = ["easy", "medium", "hard"]

# Valid action types (matches ActionType enum in models.py)
VALID_ACTION_TYPES = {"add_drug", "titrate", "remove_drug", "hold", "order_lab", "flag_physician"}

# Drug dose limits for validation & smoothing
DOSE_LIMITS: Dict[str, Tuple[float, float]] = {
    "norepinephrine": (0.01, 0.5),
    "vasopressin":    (0.01, 0.04),
    "dobutamine":     (2.0, 20.0),
    "propofol":       (5.0, 80.0),
    "fentanyl":       (25.0, 200.0),
    "insulin":        (0.5, 15.0),
}

# Tracks the last action per task for dose smoothing
_last_actions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# STRUCTURED LOGGING (OpenEnv compliance)
# =============================================================================

def _bool(val: bool) -> str:
    """Format boolean as lowercase string for OpenEnv compliance."""
    return "true" if val else "false"


def emit_start(task: str, env: str, model: str) -> None:
    """Emit [START] log line."""
    print(f"[START] task={task} env={env} model={model}")


def emit_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] log line."""
    err = error if error else "null"
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={_bool(done)} error={err}")


def emit_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={_bool(success)} steps={steps} score={score:.2f} rewards={rewards_str}")



# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert ICU clinical pharmacist AI agent managing drug titration \
for a critically ill patient in a simulated ICU.

## YOUR GOAL: Multi-System Stabilization
You must stabilize ALL organ systems — not just blood pressure. You are \
graded on ALL vitals AND ALL labs. Stabilize them SEQUENTIALLY, not all at once.

## CORE PRINCIPLE: ADAPTIVE CONTROL LOOP
You are a methodical clinician who ADAPTS based on patient response:
1. Apply ONE intervention
2. Observe the response (1–2 steps)
3. If patient is IMPROVING or STABLE → continue to HOLD
4. If patient is WORSENING → ACT immediately (break pacing)

Drug effects take time. A parameter still out of range right after adding \
a drug does NOT mean the drug failed — but if the patient is getting WORSE \
(more vitals out of range, MAP dropping further, glucose rising), you must \
respond.

## ADAPTIVE PACING RULES
After adding or titrating a drug:
- DEFAULT next action: HOLD (observe the effect)
- BREAK pacing and ACT if any of these are true:
  • Fewer vitals/labs are in range than the previous step (worsening)
  • MAP is falling further below 65 despite vasopressor
  • A life-threatening condition exists (MAP < 55, RR < 8)
  • A critical uncovered problem has no active drug addressing it

When pacing is broken:
- Address the single most critical worsening system. HIGH PRIORITY: Provide drugs for UNCOVERED systems first!
- Take ONE action, then reassess
- Resume pacing (HOLD) once the patient stabilizes

## AVOID TUNNEL VISION (CRITICAL FIX)
If a vital sign is critical (e.g., MAP < 55) but ALREADY HAS an active drug addressing it, DO NOT obsessively titrate that same drug every single step. 
You MUST step back and address other deteriorating systems (like Glucose, HR, or SpO2). 
Multi-system stability requires multi-drug concurrency. Do not wait for MAP to be perfect before adding insulin or dobutamine.

## HOLD IS SMART — BUT NOT BLIND
HOLD is correct when:
- You just added a drug AND the patient is stable or improving
- Active drugs are working and parameters are trending toward safe range
- You are unsure → HOLD is safer than guessing

DO NOT HOLD when:
- The patient is clearly worsening (more parameters out of range)
- A critical system has no drug addressing it
- Reward is decreasing (things are getting worse)

## Safe Therapeutic Ranges
### Vitals
| Parameter | Safe Range           | Priority |
|-----------|----------------------|----------|
| MAP       | 65 – 90 mmHg        | CRITICAL |
| HR        | 60 – 100 bpm        | HIGH     |
| SpO2      | 94 – 100 %          | HIGH     |
| RR        | 12 – 20 breaths/min | HIGH     |
| Temp      | 36.5 – 37.5 °C      | MODERATE |

### Labs
| Parameter   | Safe Range        | Priority |
|-------------|-------------------|----------|
| Glucose     | 70 – 180 mg/dL    | HIGH — treat with insulin |
| Creatinine  | 0.6 – 1.2 mg/dL   | MODERATE — improves with perfusion |
| Potassium   | 3.5 – 5.0 mEq/L   | HIGH — insulin lowers it |
| Lactate     | 0.5 – 2.0 mmol/L  | HIGH — improves with perfusion drugs |

## Drug-to-Problem Mapping
| Problem              | Drug to Use       | Typical Dose   | Notes |
|----------------------|-------------------|----------------|-------|
| MAP too low (<65)    | norepinephrine    | 0.05 – 0.2     | First-line vasopressor |
| MAP too low (<65)    | vasopressin       | 0.01 – 0.03    | DO NOT combine with norepinephrine |
| HR too high (>100)   | propofol          | 10 – 30        | Also lowers RR and Temp |
| HR too high (>100)   | fentanyl          | 25 – 75        | DO NOT combine with propofol |
| RR too high (>20)    | propofol          | 10 – 25        | Sedation reduces respiratory drive |
| Temp too high (>37.5)| propofol          | 10 – 25        | Mild hypothermic effect |
| SpO2 too low (<94)   | dobutamine        | 2.5 – 10       | Improves cardiac output |
| Glucose too high     | insulin           | 2 – 8          | Start 3-5 units/hr for glucose >200 |
| Potassium too high   | insulin           | 2 – 5          | Drives K+ intracellularly |
| Lactate too high     | norepinephrine    | 0.1 – 0.2      | Improved perfusion clears lactate |
| Lactate too high     | dobutamine        | 5 – 10         | Cardiac output aids clearance |
| Creatinine too high  | dobutamine        | 5 – 10         | Renal perfusion support |

## CRITICAL Safety Rules
1. NEVER combine propofol + fentanyl → respiratory depression (lethal).
2. NEVER combine vasopressin + norepinephrine → peripheral ischemia (lethal).
3. Stay within dose bounds. When titrating, change by at most 30-50%.
4. If RR drops below 10, REDUCE or REMOVE sedatives immediately.
5. AVOID FIXATION: If a problem is actively treated but another system is completely uncovered, ADD a drug for the uncovered system!

## Decision Process (FOLLOW THIS EVERY STEP)

Step A — ASSESS TREND: Is the patient improving, stable, or worsening?
  Compare current stability (vitals + labs in range) vs previous step.

Step B — CHECK PACING:
  → If you changed a drug last step AND patient is stable/improving → HOLD
  → If you changed a drug last step BUT patient is WORSENING → break pacing, proceed to Step C
  → If you did NOT change a drug last step → proceed to Step C

Step C — IDENTIFY ALL unaddressed problems.

Step D — DECIDE (pick exactly ONE action):
  0. CRITICAL RULE: If a drug is ALREADY ACTIVE, you MUST use "action_type": "titrate" to adjust it. NEVER use "add_drug" for an active drug.
  1. Life-threatening (MAP < 55 or RR < 8) → If NO drug is active for it, ADD drug. If a drug is already active for it, TITRATE it OR add a drug for another uncovered problem.
  2. Uncovered critical problem with no active drug → ADD appropriate drug. (Prioritize this over endlessly titrating existing drugs!)
  3. Drug active but system worsening → TITRATE up.
  4. All problems covered and patient stable/improving → HOLD
  5. Labs stale for 4+ steps and needed for a decision → ORDER_LAB

Step E — ACT: Execute exactly ONE action.

## IDEAL Multi-Drug Sequence (Hard Task Example)
- Step 1: Add norepinephrine → fix MAP
- Step 2: HOLD (if MAP improving) or titrate (if MAP still falling)
- Step 3: Add insulin → fix glucose/potassium
- Step 4: HOLD (if glucose responding) or titrate insulin (if not)
- Step 5: Order labs → check current values
- Step 6: Add propofol if HR/RR still high
- Step 7: HOLD → observe all systems
- Step 8: Titrate or add as needed

## Lab Ordering
- Order labs only when stale AND you need lab data for a decision.
- Every 4-5 steps is sufficient. Do NOT order every step.

## Escalation (flag_physician)
- Do NOT flag early. Only if multiple interventions have failed AND patient \
remains critically unstable after 5+ steps of active treatment.

## Response Format
Respond with ONLY a single JSON object:
{"action_type": "<type>", "drug": "<name>", "dose": <number>}

Valid action_type values:
- add_drug: Start a new drug infusion (requires drug + dose)
- titrate: Change dose of active drug (requires drug + dose)
- remove_drug: Stop a drug infusion (requires drug)
- hold: Take no action this step
- order_lab: Order fresh lab results
- flag_physician: Escalate to attending physician
"""


# =============================================================================
# OPENAI CLIENT
# =============================================================================

openai_client = OpenAI(api_key=_api_key)


# =============================================================================
# HTTP CLIENT
# =============================================================================

http_client = httpx.Client(base_url=API_BASE_URL, timeout=60.0)


# =============================================================================
# HELPER: FORMAT OBSERVATION FOR LLM
# =============================================================================

def format_observation_prompt(obs: Dict[str, Any], step: int) -> str:
    """Build a user prompt with pacing-aware system status analysis."""

    vitals = obs["vitals"]
    labs = obs["labs"]
    active_drugs = obs.get("active_drugs", [])
    alerts = obs.get("alerts", [])
    vitals_in_range = obs.get("vitals_in_range", {})
    labs_in_range = obs.get("labs_in_range", {})
    max_steps = obs.get("max_steps", MAX_STEPS)
    disease = obs.get("disease", "unknown")
    task_id = obs.get("task_id", "unknown")
    remaining = max_steps - step

    vitals_ok = sum(1 for v in vitals_in_range.values() if v)
    vitals_total = len(vitals_in_range) if vitals_in_range else 5
    labs_ok = sum(1 for v in labs_in_range.values() if v)
    labs_total = len(labs_in_range) if labs_in_range else 4

    labs_fresh = obs.get("labs_fresh", True)
    labs_status = "CURRENT" if labs_fresh else "STALE"

    lines = [
        f"== STEP {step} / {max_steps}  |  Remaining: {remaining}  |  Task: {task_id}  |  Disease: {disease} ==",
        f"   Stability: {vitals_ok}/{vitals_total} vitals OK, {labs_ok}/{labs_total} labs OK",
        "",
        "--- VITALS ---",
        f"  MAP:  {vitals['map']:.1f} mmHg       {'✓ OK' if vitals_in_range.get('map') else '✘ OUT (65-90)'}",
        f"  HR:   {vitals['hr']:.1f} bpm         {'✓ OK' if vitals_in_range.get('hr') else '✘ OUT (60-100)'}",
        f"  SpO2: {vitals['spo2']:.1f} %           {'✓ OK' if vitals_in_range.get('spo2') else '✘ OUT (94-100)'}",
        f"  RR:   {vitals['rr']:.1f} breaths/min {'✓ OK' if vitals_in_range.get('rr') else '✘ OUT (12-20)'}",
        f"  Temp: {vitals['temp']:.1f} °C          {'✓ OK' if vitals_in_range.get('temp') else '✘ OUT (36.5-37.5)'}",
        "",
        f"--- LABS [{labs_status}] ---",
        f"  Glucose:    {labs['glucose']:.1f} mg/dL   {'✓ OK' if labs_in_range.get('glucose') else '✘ OUT (70-180)'}",
        f"  Creatinine: {labs['creatinine']:.2f} mg/dL  {'✓ OK' if labs_in_range.get('creatinine') else '✘ OUT (0.6-1.2)'}",
        f"  Potassium:  {labs['potassium']:.2f} mEq/L   {'✓ OK' if labs_in_range.get('potassium') else '✘ OUT (3.5-5.0)'}",
        f"  Lactate:    {labs['lactate']:.2f} mmol/L    {'✓ OK' if labs_in_range.get('lactate') else '✘ OUT (0.5-2.0)'}",
        "",
    ]

    # Active drugs
    active_names = set()
    if active_drugs:
        lines.append("--- ACTIVE DRUGS ---")
        for d in active_drugs:
            drug_name = d['drug_name']
            active_names.add(drug_name)
            dose = d['current_dose']
            unit = d['unit']
            limits = DOSE_LIMITS.get(drug_name)
            dose_info = ""
            if limits:
                pct = ((dose - limits[0]) / (limits[1] - limits[0])) * 100 if limits[1] > limits[0] else 0
                dose_info = f"  [{pct:.0f}% of max]"
            lines.append(f"  • {drug_name}: {dose} {unit} (since step {d['step_started']}){dose_info}")
        lines.append("")
    else:
        lines.append("--- ACTIVE DRUGS: None ---")
        lines.append("")

    # === SYSTEM STATUS (calmer, non-urgent framing) ===
    needs_attention = []  # Not covered by active drugs
    being_treated = []    # Covered, waiting for effect

    # MAP
    if not vitals_in_range.get("map", False):
        if "norepinephrine" in active_names or "vasopressin" in active_names:
            being_treated.append("MAP: vasopressor active — wait for effect")
        else:
            needs_attention.append("MAP: LOW — needs vasopressor")
    # HR
    if not vitals_in_range.get("hr", False) and vitals["hr"] > 100:
        if "propofol" in active_names or "fentanyl" in active_names:
            being_treated.append("HR: sedative active — wait for effect")
        else:
            needs_attention.append("HR: HIGH — consider propofol or fentanyl")
    # SpO2
    if not vitals_in_range.get("spo2", False):
        if "dobutamine" in active_names:
            being_treated.append("SpO2: dobutamine active — wait for effect")
        else:
            needs_attention.append("SpO2: LOW — consider dobutamine")
    # RR
    if not vitals_in_range.get("rr", False) and vitals["rr"] > 20:
        if "propofol" in active_names or "fentanyl" in active_names:
            being_treated.append("RR: sedative active — wait for effect")
        else:
            needs_attention.append("RR: HIGH — consider propofol")
    # Temp
    if not vitals_in_range.get("temp", False) and vitals["temp"] > 37.5:
        if "propofol" in active_names:
            being_treated.append("Temp: propofol active — wait for effect")
        else:
            needs_attention.append("Temp: HIGH — propofol may help")
    # Glucose
    if not labs_in_range.get("glucose", False) and labs["glucose"] > 180:
        if "insulin" in active_names:
            being_treated.append("Glucose: insulin active — wait for effect")
        else:
            needs_attention.append("Glucose: HIGH — needs insulin")
    # Potassium
    if not labs_in_range.get("potassium", False) and labs["potassium"] > 5.0:
        if "insulin" in active_names:
            being_treated.append("K+: insulin active — wait for effect")
        else:
            needs_attention.append("K+: HIGH — insulin helps")
    # Lactate
    if not labs_in_range.get("lactate", False):
        if "dobutamine" in active_names or "norepinephrine" in active_names:
            being_treated.append("Lactate: perfusion drugs active — wait for effect")
        else:
            needs_attention.append("Lactate: HIGH — needs perfusion support")

    if being_treated:
        lines.append(f"--- BEING TREATED ({len(being_treated)}) — wait for drug effect ---")
        for item in being_treated:
            lines.append(f"  ✓ {item}")
        lines.append("")

    if needs_attention:
        lines.append(f"--- NEEDS ATTENTION ({len(needs_attention)}) ---")
        for item in needs_attention:
            lines.append(f"  • {item}")
        lines.append("")

    # Alerts
    if alerts:
        lines.append("--- ⚠ ALERTS ---")
        for a in alerts:
            severity = a.get("severity", "info").upper()
            lines.append(f"  [{severity}] {a['message']}")
        lines.append("")

    # === ADAPTIVE PACING DECISION GUIDANCE ===
    lines.append("─" * 50)

    # Detect if a drug was recently changed
    recently_acted = False
    last_action_type = ""
    if _action_history:
        last_action = _action_history[-1].get("action", {})
        last_action_type = last_action.get("action_type", "")
        if last_action_type in ("add_drug", "titrate", "remove_drug"):
            recently_acted = True

    # Detect patient TREND by comparing to previous step
    current_score = vitals_ok + labs_ok
    prev_score = None
    trend = "unknown"
    if len(_action_history) >= 1:
        prev_score = _action_history[-1].get("score", None)
    if prev_score is not None:
        if current_score > prev_score:
            trend = "improving"
        elif current_score == prev_score:
            trend = "stable"
        else:
            trend = "worsening"

    # Life-threatening check
    life_threat = vitals["map"] < 55 or vitals.get("rr", 16) < 8

    # Show trend to the LLM
    if trend == "improving":
        lines.append(f"TREND: IMPROVING ({prev_score}/9 → {current_score}/9)")
    elif trend == "worsening":
        lines.append(f"TREND: ⚠ WORSENING ({prev_score}/9 → {current_score}/9)")
    elif trend == "stable" and prev_score is not None:
        lines.append(f"TREND: STABLE ({current_score}/9)")
    else:
        lines.append(f"STATUS: {current_score}/9 systems in range")

    # Adaptive decision guidance
    if life_threat:
        lines.append("⚠ LIFE-THREATENING — act immediately. BUT AVOID TUNNEL VISION!")
        lines.append("If the life-threatening system already has an active drug, ADD medication for OTHER uncovered systems concurrently!")
    elif trend == "worsening" and recently_acted:
        lines.append("Patient is WORSENING despite recent drug change. Break pacing — ACT now.")
        lines.append("Identify the most critical deteriorating system and intervene.")
    elif trend == "worsening" and needs_attention:
        lines.append(f"Patient is WORSENING. Address the most critical uncovered problem.")
        lines.append("Take ONE action to stabilize, then reassess.")
    elif recently_acted and trend in ("improving", "stable", "unknown"):
        lines.append("Drug changed last step and patient is stable/improving. HOLD to observe.")
    elif vitals_ok + labs_ok == vitals_total + labs_total:
        lines.append("ALL SYSTEMS STABLE → HOLD")
    elif needs_attention:
        lines.append(f"{len(needs_attention)} system(s) need attention. Address the most critical one.")
        lines.append("Then HOLD next step to observe the effect.")
    else:
        lines.append("All out-of-range systems have active drugs. HOLD and let them work.")

    if not labs_fresh and step > 0 and not recently_acted and not trend == "worsening":
        if step % 4 == 0 or step % 5 == 0:
            lines.append("Labs are stale. Consider ordering labs if not acting on a vital.")

    if remaining <= 3:
        lines.append("FEW STEPS REMAINING — prefer stability over new interventions.")
    lines.append("")
    lines.append("Respond with ONLY a JSON object.")

    return "\n".join(lines)


# =============================================================================
# STABILITY HEURISTIC
# =============================================================================

def stability_heuristic(obs: Dict[str, Any]) -> bool:
    """
    Check whether the patient is currently stable enough to HOLD.

    Returns True if all critical vitals are within safe ranges,
    meaning no intervention is needed this step.
    """
    vitals = obs.get("vitals", {})
    vitals_in_range = obs.get("vitals_in_range", {})

    # Primary vitals that must be in range to consider patient "stable"
    map_ok = vitals_in_range.get("map", False)
    hr_ok = vitals_in_range.get("hr", False)
    spo2_ok = vitals_in_range.get("spo2", False)
    rr_ok = vitals_in_range.get("rr", False)

    # Use tighter MAP range for stability (65–85) to leave buffer
    map_val = vitals.get("map", 0)
    map_stable = 65.0 <= map_val <= 85.0

    return map_stable and map_ok and hr_ok and spo2_ok and rr_ok


# =============================================================================
# DOSE SMOOTHING
# =============================================================================

def smooth_dose(action: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dose smoothing to prevent large oscillations.

    If the agent is titrating an active drug, limit the dose change to at
    most 50% of the current dose per step. Also clamps to valid dose bounds.
    """
    action = dict(action)  # copy
    action_type = action.get("action_type")
    drug = action.get("drug")
    new_dose = action.get("dose")

    # ── AUTO-CORRECT: Convert add_drug to titrate for active drugs ──
    active_drugs = obs.get("active_drugs", [])
    active_names = {d["drug_name"] for d in active_drugs}
    
    if action_type == "add_drug" and drug in active_names:
        action["action_type"] = "titrate"
        action_type = "titrate"

    if action_type != "titrate" or drug is None or new_dose is None:
        return action

    # Find current dose of this drug
    active_drugs = obs.get("active_drugs", [])
    current_dose = None
    for d in active_drugs:
        if d["drug_name"] == drug:
            current_dose = d["current_dose"]
            break

    if current_dose is not None and current_dose > 0:
        max_change = current_dose * 0.5  # Max 50% change per step
        delta = new_dose - current_dose
        if abs(delta) > max_change:
            clamped_dose = current_dose + (max_change if delta > 0 else -max_change)
            new_dose = clamped_dose

    # Clamp to valid dose bounds
    if drug in DOSE_LIMITS:
        lo, hi = DOSE_LIMITS[drug]
        new_dose = max(lo, min(hi, new_dose))

    action = dict(action)  # copy
    action["dose"] = round(new_dose, 4)
    return action


# Action history for conversation memory (per-task, reset in run_task)
_action_history: List[Dict[str, Any]] = []


def call_llm(obs: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
    """
    Send the current observation to the LLM and return the parsed action dict.

    The LLM is ALWAYS called (no pre-LLM heuristic bypass) to ensure
    the benchmark measures actual LLM clinical reasoning.

    Includes action history from previous steps so the LLM has memory of
    its prior decisions and doesn't repeat the same single-drug behavior.

    Returns None if the LLM call or parsing fails entirely.
    """

    user_prompt = format_observation_prompt(obs, step)

    # Build messages with action history for memory
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add last 3 action history entries as context
    for hist in _action_history[-3:]:
        messages.append({"role": "user", "content": hist["obs_summary"]})
        messages.append({"role": "assistant", "content": json.dumps(hist["action"])})

    # Current observation
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,  # Slight diversity to explore different drugs
            max_tokens=256,
        )
        raw = response.choices[0].message.content
        if raw is None:
            return None
        action = parse_action(raw.strip())
        if action is None:
            return None
        # Apply dose smoothing to prevent oscillations
        action = smooth_dose(action, obs)

        # Record for history (including score for trend detection)
        vitals_ok = sum(1 for v in obs.get("vitals_in_range", {}).values() if v)
        labs_ok = sum(1 for v in obs.get("labs_in_range", {}).values() if v)
        score = vitals_ok + labs_ok
        obs_summary = f"Step {step}: {vitals_ok}/5 vitals OK, {labs_ok}/4 labs OK"
        _action_history.append({"obs_summary": obs_summary, "action": action, "score": score})

        return action
    except Exception as exc:
        sys.stderr.write(f"LLM error: {exc}\n")
        return None


# =============================================================================
# PARSE ACTION
# =============================================================================

def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON action from raw LLM output.

    Handles:
        - Plain JSON
        - JSON wrapped in ```json ... ``` code fences
        - Multiple JSON objects (takes the first valid one)
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if "```" in cleaned:
        # Extract content between code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    # Try direct parse
    try:
        action = json.loads(cleaned)
        return _validate_action(action)
    except json.JSONDecodeError:
        pass

    # Try to find first JSON object in the string
    match = re.search(r"\{[^{}]*\}", cleaned)
    if match:
        try:
            action = json.loads(match.group(0))
            return _validate_action(action)
        except json.JSONDecodeError:
            pass

    return None


def _validate_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalise an action dict. Returns None if invalid."""
    if not isinstance(action, dict):
        return None

    action_type = action.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        return None

    # Build a clean action dict with only known fields
    clean: Dict[str, Any] = {"action_type": action_type}

    if action_type in ("add_drug", "titrate", "remove_drug"):
        drug = action.get("drug")
        if not drug:
            return None
        clean["drug"] = str(drug).lower()

    if action_type in ("add_drug", "titrate"):
        dose = action.get("dose")
        if dose is None:
            return None
        try:
            clean["dose"] = float(dose)
        except (TypeError, ValueError):
            return None

    return clean


# =============================================================================
# FALLBACK POLICY
# =============================================================================

def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback policy when the LLM fails.

    Multi-system strategy (evaluated in priority order):
        1. If MAP < 65 → titrate/add norepinephrine
        2. If RR < 10  → reduce or remove sedatives
        3. If glucose > 200 and no insulin → add insulin
        4. If SpO2 < 94 and no dobutamine → add dobutamine
        5. If HR > 110 and no sedative → add propofol at low dose
        6. If labs are stale → order labs
        7. Otherwise → hold
    """
    vitals = obs.get("vitals", {})
    labs = obs.get("labs", {})
    map_val = vitals.get("map", 75.0)
    rr_val = vitals.get("rr", 16.0)
    hr_val = vitals.get("hr", 80.0)
    spo2_val = vitals.get("spo2", 97.0)
    glucose_val = labs.get("glucose", 120.0)
    active_drugs = obs.get("active_drugs", [])
    labs_fresh = obs.get("labs_fresh", True)

    active_names = {d["drug_name"] for d in active_drugs}

    # ── Priority 1: Respiratory depression ─────────────────────
    if rr_val < 10.0:
        for d in active_drugs:
            if d["drug_name"] in ("propofol", "fentanyl"):
                current = d["current_dose"]
                lo = DOSE_LIMITS.get(d["drug_name"], (0, 0))[0]
                if current <= lo * 1.5:
                    return {"action_type": "remove_drug", "drug": d["drug_name"]}
                new_dose = round(max(lo, current * 0.7), 4)
                return {"action_type": "titrate", "drug": d["drug_name"], "dose": new_dose}

    # ── Priority 2: Uncovered Hypotension ────────────────────────────────
    if map_val < 65.0 and "norepinephrine" not in active_names:
        return {"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1}

    # ── Priority 3: Uncovered Hyperglycemia ──────────────────────────────
    if glucose_val > 200.0 and "insulin" not in active_names:
        return {"action_type": "add_drug", "drug": "insulin", "dose": 4.0}

    # ── Priority 4: Uncovered Hypoxia ────────────────────────────────────
    if spo2_val < 94.0 and "dobutamine" not in active_names:
        return {"action_type": "add_drug", "drug": "dobutamine", "dose": 5.0}

    # ── Priority 5: Uncovered Tachycardia ────────────────────────────────
    if hr_val > 110.0 and "propofol" not in active_names and "fentanyl" not in active_names:
        return {"action_type": "add_drug", "drug": "propofol", "dose": 15.0}

    # ── Priority 6: Titrate existing vasopressor if still crashing ──────────────
    if map_val < 65.0:
        norepi_dose = None
        for d in active_drugs:
            if d["drug_name"] == "norepinephrine":
                norepi_dose = d["current_dose"]
                break

        if norepi_dose is not None:
            increment = min(norepi_dose * 0.3, 0.05)
            new_dose = min(0.5, round(norepi_dose + increment, 4))
            return {"action_type": "titrate", "drug": "norepinephrine", "dose": new_dose}

    # ── Priority 7: Stale labs ─────────────────────────────────
    if not labs_fresh:
        return {"action_type": "order_lab"}

    # ── Default: Hold ──────────────────────────────────────────
    return {"action_type": "hold"}


# =============================================================================
# RUN SINGLE TASK
# =============================================================================

def run_task(task_id: str) -> Dict[str, Any]:
    """
    Execute a single task (easy / medium / hard) against the environment.

    Returns a results dict with score, total_reward, steps, and breakdown.
    """
    global _action_history
    _action_history = []  # Reset action memory for new task

    session_id = f"inference-{task_id}-{uuid.uuid4().hex[:8]}"

    # ── Reset ──────────────────────────────────────────────────
    try:
        res = http_client.post(
            "/reset",
            json={"task_id": task_id},
            params={"session_id": session_id},
        )
        res.raise_for_status()
        data = res.json()
    except Exception as exc:
        return {"score": 0.0, "total_reward": 0.0, "steps": 0, "error": str(exc)}

    obs = data["observation"]
    session_id = data.get("session_id", session_id)
    total_reward = 0.0
    step = 0
    done = False
    step_rewards: List[float] = []

    emit_start(task_id, "icu_drug_titration", MODEL_NAME)

    # ── Step Loop ──────────────────────────────────────────────
    while not done and step < MAX_STEPS:
        current_step = obs.get("current_step", step)

        # Get action from LLM (wrapped — never crashes)
        try:
            action = call_llm(obs, current_step)
        except Exception as exc:
            sys.stderr.write(f"LLM call_llm error: {exc}\n")
            action = None

        # Fall back if LLM returned nothing valid
        if action is None:
            action = fallback_action(obs)

        # Execute step
        try:
            res = http_client.post(
                "/step",
                json=action,
                params={"session_id": session_id},
            )
            res.raise_for_status()
            step_data = res.json()
        except Exception as exc:
            sys.stderr.write(f"Step API error: {exc}\n")
            # On API error, try fallback then retry once
            action = fallback_action(obs)
            try:
                res = http_client.post(
                    "/step",
                    json=action,
                    params={"session_id": session_id},
                )
                res.raise_for_status()
                step_data = res.json()
            except Exception as exc2:
                sys.stderr.write(f"Step API retry failed: {exc2}\n")
                step += 1
                emit_step(step, json.dumps(action), 0.0, False, str(exc2))
                break

        # Parse step data safely
        try:
            reward_val = step_data["reward"]["value"]
        except (KeyError, TypeError):
            reward_val = 0.0
        total_reward += reward_val
        step_rewards.append(reward_val)
        obs = step_data.get("observation", obs)  # keep last obs on failure
        done = step_data.get("done", False)
        step += 1

        emit_step(step, json.dumps(action), reward_val, done, None)

    # ── Grade ──────────────────────────────────────────────────
    score = 0.0
    breakdown: Dict[str, Any] = {}
    done_reason: Optional[str] = None

    try:
        res = http_client.get("/grader", params={"session_id": session_id})
        res.raise_for_status()
        grade = res.json()
        score = grade["score"]
        breakdown = grade.get("breakdown", {})
        done_reason = grade.get("done_reason")
    except Exception as exc:
        pass  # Grader error — score stays 0.0

    emit_end(score > 0.0, step, score, step_rewards)

    return {
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "done_reason": done_reason,
        "breakdown": breakdown,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the LLM inference agent across all three tasks."""
    # Verify server is reachable (warn but do NOT exit)
    try:
        health = http_client.get("/health")
        health.raise_for_status()
    except Exception as exc:
        sys.stderr.write(f"WARNING: Cannot reach server at {API_BASE_URL}: {exc}\n")
        sys.stderr.write("Will attempt tasks anyway (may fail individually).\n")

    results: Dict[str, Dict[str, Any]] = {}

    for task_id in TASKS:
        try:
            results[task_id] = run_task(task_id)
        except Exception as exc:
            sys.stderr.write(f"Task {task_id} crashed unexpectedly: {exc}\n")
            results[task_id] = {"score": 0.0, "total_reward": 0.0, "steps": 0, "error": str(exc)}

    # Report average score but NEVER exit non-zero due to low score
    avg_score = sum(r["score"] for r in results.values()) / len(TASKS) if TASKS else 0.0
    sys.stderr.write(f"Average score: {avg_score:.4f}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"Fatal error in main: {exc}\n")
        # Exit 0 -- inference must never crash with non-zero
        sys.exit(0)
