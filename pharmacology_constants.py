"""
Pharmacology Constants for ICU Drug Titration Environment.

Contains all deterministic parameters for drug effects, vital sign ranges,
lab value ranges, disease profiles, drug-drug interactions, and biological
noise parameters. These constants form the foundation of the simulation engine.

Drug effect multipliers use an EQUILIBRIUM (offset) model:
  effect_offset = dose × multiplier
The offset is applied as a steady-state contribution each step (not accumulated).
When a drug is removed, the offset disappears immediately.
"""

from typing import Dict, List, Tuple

# =============================================================================
# DRUG DEFINITIONS
# =============================================================================

# Drug categories
VASOPRESSORS = ["norepinephrine", "vasopressin"]
INOTROPES = ["dobutamine"]
SEDATIVES = ["propofol"]
ANALGESICS = ["fentanyl"]
METABOLIC = ["insulin"]

ALL_DRUGS = VASOPRESSORS + INOTROPES + SEDATIVES + ANALGESICS + METABOLIC

# Drug dose ranges: (min_dose, max_dose, unit)
DRUG_DOSE_RANGES: Dict[str, Tuple[float, float, str]] = {
    "norepinephrine": (0.01, 0.5, "mcg/kg/min"),
    "vasopressin":    (0.01, 0.04, "units/min"),
    "dobutamine":     (2.0, 20.0, "mcg/kg/min"),
    "propofol":       (5.0, 80.0, "mcg/kg/min"),
    "fentanyl":       (25.0, 200.0, "mcg/hr"),
    "insulin":        (0.5, 15.0, "units/hr"),
}

# =============================================================================
# DRUG EFFECT MULTIPLIERS (EQUILIBRIUM / OFFSET MODEL)
# Each drug has a dict mapping vital/lab → steady-state offset per unit dose.
# Effect offset = dose × multiplier (applied as offset from undrugged state).
# =============================================================================

DRUG_EFFECTS: Dict[str, Dict[str, float]] = {
    "norepinephrine": {
        "map":  300.0,    # dose 0.1 → +30 mmHg offset
        "hr":   80.0,     # dose 0.1 → +8 bpm (reflex tachycardia)
        "spo2": 0.0,
        "rr":   0.0,
        "temp": 0.0,
        "glucose": 0.0,
        "creatinine": 0.0,
        "potassium": 0.0,
        "lactate": -3.0,  # dose 0.1 → -0.3 (improved perfusion)
    },
    "vasopressin": {
        "map":  1000.0,   # dose 0.02 → +20 mmHg offset
        "hr":   -350.0,   # dose 0.02 → -7 bpm (reflex bradycardia)
        "spo2": 0.0,
        "rr":   0.0,
        "temp": 0.0,
        "glucose": 0.0,
        "creatinine": 0.0,
        "potassium": 0.0,
        "lactate": -15.0, # dose 0.02 → -0.3
    },
    "dobutamine": {
        "map":  1.5,      # dose 5 → +7.5 mmHg
        "hr":   2.5,      # dose 5 → +12.5 bpm
        "spo2": 0.5,      # dose 5 → +2.5% (improved cardiac output)
        "rr":   0.0,
        "temp": 0.06,     # dose 5 → +0.3°C (thermogenic)
        "glucose": 0.0,
        "creatinine": -0.01,  # dose 5 → -0.05 (improved renal perfusion)
        "potassium": -0.02,
        "lactate": -0.08,
    },
    "propofol": {
        "map":  -0.35,    # dose 20 → -7 mmHg
        "hr":   -0.5,     # dose 20 → -10 bpm
        "spo2": -0.07,    # dose 20 → -1.4%
        "rr":   -0.25,    # dose 20 → -5 breaths/min
        "temp": -0.015,   # dose 20 → -0.3°C
        "glucose": 0.0,
        "creatinine": 0.0,
        "potassium": 0.0,
        "lactate": 0.0,
    },
    "fentanyl": {
        "map":  -0.04,    # dose 50 → -2 mmHg
        "hr":   -0.08,    # dose 50 → -4 bpm
        "spo2": -0.01,    # dose 50 → -0.5%
        "rr":   -0.07,    # dose 50 → -3.5 breaths/min
        "temp": 0.0,
        "glucose": 0.0,
        "creatinine": 0.0,
        "potassium": 0.0,
        "lactate": 0.0,
    },
    "insulin": {
        "map":  0.0,
        "hr":   0.0,
        "spo2": 0.0,
        "rr":   0.0,
        "temp": 0.0,
        "glucose": -15.0,   # dose 3 → -45 mg/dL
        "creatinine": 0.0,
        "potassium": -0.1,  # dose 3 → -0.3 mEq/L (intracellular shift)
        "lactate": 0.0,
    },
}

# =============================================================================
# VITAL SIGN RANGES
# =============================================================================

# Safe (target) ranges for vitals
VITAL_SAFE_RANGES: Dict[str, Tuple[float, float]] = {
    "map":  (65.0, 90.0),    # mmHg
    "hr":   (60.0, 100.0),   # bpm
    "spo2": (94.0, 100.0),   # %
    "rr":   (12.0, 20.0),    # breaths/min
    "temp": (36.5, 37.5),    # °C
}

# Physiological clamp ranges (absolute limits)
VITAL_CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
    "map":  (20.0, 200.0),
    "hr":   (20.0, 220.0),
    "spo2": (50.0, 100.0),
    "rr":   (4.0, 45.0),
    "temp": (32.0, 42.0),
}

# Critical (lethal) thresholds — crossing these ends the episode
VITAL_CRITICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "map":  (30.0, 180.0),
    "hr":   (25.0, 200.0),
    "spo2": (60.0, 100.0),
    "rr":   (5.0, 40.0),
    "temp": (33.0, 41.0),
}

# =============================================================================
# LAB VALUE RANGES
# =============================================================================

LAB_SAFE_RANGES: Dict[str, Tuple[float, float]] = {
    "glucose":    (70.0, 180.0),    # mg/dL
    "creatinine": (0.6, 1.2),      # mg/dL
    "potassium":  (3.5, 5.0),      # mEq/L
    "lactate":    (0.5, 2.0),      # mmol/L
}

LAB_CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
    "glucose":    (20.0, 600.0),
    "creatinine": (0.2, 15.0),
    "potassium":  (2.0, 8.0),
    "lactate":    (0.1, 20.0),
}

# =============================================================================
# DISEASE PROFILES
# =============================================================================

class DiseaseProfile:
    """Defines a disease condition with baseline vitals, labs, and deterioration rates."""

    def __init__(
        self,
        name: str,
        description: str,
        baseline_vitals: Dict[str, float],
        baseline_labs: Dict[str, float],
        deterioration_per_hour: Dict[str, float],
        allowed_drugs: List[str],
    ):
        self.name = name
        self.description = description
        self.baseline_vitals = baseline_vitals
        self.baseline_labs = baseline_labs
        self.deterioration_per_hour = deterioration_per_hour
        self.allowed_drugs = allowed_drugs


DISEASE_PROFILES: Dict[str, DiseaseProfile] = {
    "vasopressor_shock": DiseaseProfile(
        name="vasopressor_shock",
        description="Distributive shock requiring vasopressor support to maintain MAP",
        baseline_vitals={
            "map": 52.0,   # Hypotensive
            "hr": 110.0,   # Tachycardic
            "spo2": 96.0,
            "rr": 22.0,    # Mildly tachypneic
            "temp": 37.8,  # Low-grade fever
        },
        baseline_labs={
            "glucose": 160.0,
            "creatinine": 1.0,
            "potassium": 4.2,
            "lactate": 3.5,  # Elevated from poor perfusion
        },
        deterioration_per_hour={
            "map": -1.5,    # MAP drops without treatment
            "hr": 1.0,      # HR rises compensatory
            "spo2": -0.2,
            "rr": 0.3,
            "temp": 0.05,
            "glucose": 2.0,
            "creatinine": 0.02,
            "potassium": 0.03,
            "lactate": 0.3,
        },
        allowed_drugs=["norepinephrine", "vasopressin"],
    ),
    "ventilated_sedation": DiseaseProfile(
        name="ventilated_sedation",
        description="Mechanically ventilated patient requiring sedation and analgesia management",
        baseline_vitals={
            "map": 78.0,
            "hr": 88.0,
            "spo2": 97.0,
            "rr": 18.0,
            "temp": 37.2,
        },
        baseline_labs={
            "glucose": 145.0,
            "creatinine": 0.9,
            "potassium": 4.0,
            "lactate": 1.5,
        },
        deterioration_per_hour={
            "map": 0.5,     # Slight hypertension from agitation
            "hr": 1.5,      # HR rising from agitation
            "spo2": -0.1,
            "rr": 0.8,      # RR rising from agitation/pain
            "temp": 0.03,
            "glucose": 3.0,
            "creatinine": 0.01,
            "potassium": 0.02,
            "lactate": 0.1,
        },
        allowed_drugs=["norepinephrine", "vasopressin", "dobutamine", "propofol", "fentanyl", "insulin"],
    ),
    "septic_renal_failure": DiseaseProfile(
        name="septic_renal_failure",
        description="Septic shock with acute kidney injury requiring multi-drug management",
        baseline_vitals={
            "map": 48.0,    # Severely hypotensive
            "hr": 120.0,    # Significant tachycardia
            "spo2": 93.0,   # Borderline oxygenation
            "rr": 26.0,     # Tachypneic
            "temp": 39.0,   # Febrile
        },
        baseline_labs={
            "glucose": 220.0,   # Stress hyperglycemia
            "creatinine": 2.8,  # Acute kidney injury
            "potassium": 5.5,   # Hyperkalemia from renal failure
            "lactate": 5.0,     # Significantly elevated
        },
        deterioration_per_hour={
            "map": -2.0,     # Rapid MAP decline
            "hr": 1.5,
            "spo2": -0.4,
            "rr": 0.5,
            "temp": 0.1,
            "glucose": 5.0,    # Worsening hyperglycemia
            "creatinine": 0.08, # Worsening renal function
            "potassium": 0.08,  # Rising potassium
            "lactate": 0.5,     # Worsening tissue perfusion
        },
        allowed_drugs=["norepinephrine", "vasopressin", "dobutamine", "propofol", "fentanyl", "insulin"],
    ),
}

# =============================================================================
# DRUG-DRUG INTERACTIONS (EQUILIBRIUM OFFSET MODEL)
# Effects are applied as steady-state offsets while both drugs are active.
# =============================================================================

# Critical interactions: severe adverse effects, immediate danger
# Format: frozenset({drug1, drug2}): (severity, description, affected_vital, effect_offset)
CRITICAL_INTERACTIONS: Dict[frozenset, Tuple[str, str, str, float]] = {
    frozenset({"propofol", "fentanyl"}): (
        "critical",
        "Respiratory depression: concurrent sedative + opioid synergy",
        "rr",
        -4.0,  # RR offset while both active
    ),
    frozenset({"vasopressin", "norepinephrine"}): (
        "critical",
        "Peripheral ischemia risk: dual vasopressor synergy",
        "map",
        15.0,  # Excessive MAP spike while both active
    ),
}

# Warning interactions: moderate risk, should be monitored
WARNING_INTERACTIONS: Dict[frozenset, Tuple[str, str, str, float]] = {
    frozenset({"dobutamine", "norepinephrine"}): (
        "warning",
        "Tachycardia risk: combined chronotropic effects",
        "hr",
        8.0,  # HR offset while both active
    ),
    frozenset({"propofol", "norepinephrine"}): (
        "warning",
        "Hemodynamic instability: opposing MAP effects",
        "map",
        -5.0,  # Partially counteract vasopressor
    ),
    frozenset({"insulin", "propofol"}): (
        "warning",
        "Hypoglycemia risk: propofol lipid emulsion alters glucose metabolism",
        "glucose",
        -15.0,  # Enhanced glucose lowering
    ),
}

# =============================================================================
# BIOLOGICAL NOISE PARAMETERS (Gaussian σ per vital/lab)
# =============================================================================

NOISE_SIGMA: Dict[str, float] = {
    "map":        1.5,
    "hr":         1.0,
    "spo2":       0.3,
    "rr":         0.5,
    "temp":       0.05,
    "glucose":    3.0,
    "creatinine": 0.02,
    "potassium":  0.05,
    "lactate":    0.1,
}

# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASK_DEFINITIONS: Dict[str, Dict] = {
    "easy": {
        "name": "MAP Control with Norepinephrine",
        "description": "Control MAP using norepinephrine in a vasopressor shock patient. "
                       "Goal: maintain MAP in safe range (65-90 mmHg).",
        "disease": "vasopressor_shock",
        "horizon": 12,
        "allowed_drugs": ["norepinephrine"],
        "primary_vitals": ["map"],
        "difficulty": "easy",
    },
    "medium": {
        "name": "Multi-Drug Ventilated Sedation",
        "description": "Manage a ventilated patient with sedation, analgesia, and hemodynamic "
                       "support. Balance 3+ drugs with interaction risks.",
        "disease": "ventilated_sedation",
        "horizon": 20,
        "allowed_drugs": ["norepinephrine", "propofol", "fentanyl"],
        "primary_vitals": ["map", "hr", "rr"],
        "difficulty": "medium",
    },
    "hard": {
        "name": "Septic Shock with Renal Failure",
        "description": "Manage a septic shock patient with acute kidney injury. Multi-drug "
                       "titration with renal constraints, electrolyte management, and glucose control.",
        "disease": "septic_renal_failure",
        "horizon": 24,
        "allowed_drugs": ["norepinephrine", "vasopressin", "dobutamine", "propofol", "fentanyl", "insulin"],
        "primary_vitals": ["map", "hr", "spo2", "rr", "temp"],
        "difficulty": "hard",
    },
}

# =============================================================================
# REWARD CONSTANTS
# =============================================================================

REWARD_VITAL_IN_RANGE = 0.1       # Per vital in safe range per step
REWARD_LAB_IN_RANGE = 0.1         # Per lab in safe range per step (raised to match vitals)
REWARD_CRITICAL_INTERACTION = -1.0 # Critical drug-drug interaction (base, escalates)
REWARD_WARNING_INTERACTION = -0.3  # Warning drug-drug interaction (base, escalates)
REWARD_UNSAFE_DOSE = -0.5         # Dose outside safe bounds
REWARD_INVALID_ACTION = -0.2      # Invalid/malformed action penalty
REWARD_MULTI_COVERAGE_BONUS = 0.15 # Bonus when ≥3 vitals + ≥2 labs in range
REWARD_DETERIORATION_PENALTY = -0.05  # Per vital/lab OUT of range (pressure to fix all systems)
REWARD_TERMINAL_BONUS = 2.0       # All vitals stable at episode end
REWARD_FLAG_PHYSICIAN = -1.5      # Giving up penalty
REWARD_PATIENT_DEATH = -5.0       # Patient death penalty
