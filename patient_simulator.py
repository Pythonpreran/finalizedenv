"""
Patient Simulator for ICU Drug Titration Environment.

Implements EQUILIBRIUM-based pharmacology simulation with:
- Disease-specific baseline physiology
- Steady-state drug effect offsets (NOT accumulating)
- Drug-drug interaction detection and offset effects
- Lab visibility gated behind order_lab action
- Seeded Gaussian biological noise
- Dose validation and safety checking

Drug Model:
  Each step, vitals are computed as:
    vitals = undrugged_state + Σ(drug_offsets) + Σ(interaction_offsets) + noise
  Where undrugged_state = baseline + accumulated_deterioration.
  When a drug is removed, its offset disappears immediately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pharmacology_constants import (
    CRITICAL_INTERACTIONS,
    DISEASE_PROFILES,
    DRUG_DOSE_RANGES,
    DRUG_EFFECTS,
    LAB_CLAMP_RANGES,
    LAB_SAFE_RANGES,
    NOISE_SIGMA,
    VITAL_CLAMP_RANGES,
    VITAL_CRITICAL_RANGES,
    VITAL_SAFE_RANGES,
    WARNING_INTERACTIONS,
)
from models import (
    Action,
    ActionType,
    Alert,
    DrugInfo,
    Labs,
    Vitals,
)


class PatientSimulator:
    """
    Simulates ICU patient physiology with equilibrium-based pharmacology.

    The simulator maintains an "undrugged" baseline that deteriorates over time.
    Drug effects are applied as steady-state offsets (not accumulated),
    ensuring that stopping a drug removes its effect immediately.

    Args:
        disease_name: Name of the disease profile to simulate.
        seed: Random seed for biological noise reproducibility.
    """

    def __init__(self, disease_name: str, seed: int = 42):
        if disease_name not in DISEASE_PROFILES:
            raise ValueError(
                f"Unknown disease: {disease_name}. "
                f"Available: {list(DISEASE_PROFILES.keys())}"
            )
        self.disease_name = disease_name
        self.disease = DISEASE_PROFILES[disease_name]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Current patient state
        self.vitals: Dict[str, float] = {}
        self.labs: Dict[str, float] = {}
        # Undrugged state: baseline + accumulated deterioration
        self._undrugged_vitals: Dict[str, float] = {}
        self._undrugged_labs: Dict[str, float] = {}
        # Visible labs (gated behind order_lab)
        self._visible_labs: Dict[str, float] = {}
        self._labs_fresh: bool = True  # True on reset (initial admission labs)

        self.active_drugs: Dict[str, DrugInfo] = {}
        self.current_step: int = 0
        self.alerts: List[Alert] = []
        self._had_invalid_action: bool = False

    def reset(self) -> Tuple[Vitals, Labs]:
        """
        Reset the patient to baseline disease state.

        Returns:
            Tuple of (Vitals, Labs) representing initial patient state.
        """
        self.rng = np.random.RandomState(self.seed)
        self.current_step = 0
        self.active_drugs = {}
        self.alerts = []
        self._had_invalid_action = False

        # Initialize undrugged state from disease baseline
        self._undrugged_vitals = dict(self.disease.baseline_vitals)
        self._undrugged_labs = dict(self.disease.baseline_labs)

        # Current state starts at undrugged baseline
        self.vitals = dict(self._undrugged_vitals)
        self.labs = dict(self._undrugged_labs)

        # Initial admission labs are visible
        self._visible_labs = dict(self.labs)
        self._labs_fresh = True

        return self._make_vitals(), self._make_labs()

    def apply_action(
        self,
        action: Action,
        allowed_drugs: List[str],
        step: int,
    ) -> Tuple[Vitals, Labs, List[Alert], bool, str]:
        """
        Apply a clinical action and advance the simulation by one hour.

        Uses equilibrium model: vitals = undrugged + drug_offsets + interactions + noise.

        Args:
            action: The clinical action to execute.
            allowed_drugs: List of drugs permitted for this task.
            step: Current simulation step.

        Returns:
            Tuple of (vitals, labs, alerts, is_fatal, fatal_reason)
        """
        self.current_step = step
        self.alerts = []
        self._had_invalid_action = False
        is_fatal = False
        fatal_reason = ""
        labs_ordered = False

        # --- Process the action ---
        if action.action_type == ActionType.ADD_DRUG:
            self._handle_add_drug(action, allowed_drugs)
        elif action.action_type == ActionType.TITRATE:
            self._handle_titrate(action, allowed_drugs)
        elif action.action_type == ActionType.REMOVE_DRUG:
            self._handle_remove_drug(action)
        elif action.action_type == ActionType.ORDER_LAB:
            labs_ordered = True
            self.alerts.append(Alert(
                severity="info",
                message="Lab panel ordered — results updated",
                source="lab_order",
            ))
        elif action.action_type == ActionType.HOLD:
            pass  # No drug changes
        # FLAG_PHYSICIAN is handled at the environment level

        # --- Apply disease deterioration to undrugged state ---
        for vital, rate in self.disease.deterioration_per_hour.items():
            if vital in self._undrugged_vitals:
                self._undrugged_vitals[vital] += rate
            elif vital in self._undrugged_labs:
                self._undrugged_labs[vital] += rate

        # --- Compute vitals from undrugged + drug offsets (EQUILIBRIUM) ---
        self.vitals = dict(self._undrugged_vitals)
        self.labs = dict(self._undrugged_labs)

        # Apply drug effect offsets
        for drug_name, drug_info in self.active_drugs.items():
            if drug_name in DRUG_EFFECTS:
                effects = DRUG_EFFECTS[drug_name]
                dose = drug_info.current_dose
                for param, multiplier in effects.items():
                    effect = dose * multiplier
                    if param in self.vitals:
                        self.vitals[param] += effect
                    elif param in self.labs:
                        self.labs[param] += effect

        # --- Apply drug-drug interaction offsets ---
        self._apply_interactions()

        # --- Apply biological noise ---
        for vital in self.vitals:
            if vital in NOISE_SIGMA:
                noise = self.rng.normal(0, NOISE_SIGMA[vital])
                self.vitals[vital] += noise

        for lab in self.labs:
            if lab in NOISE_SIGMA:
                noise = self.rng.normal(0, NOISE_SIGMA[lab])
                self.labs[lab] += noise

        # --- Clamp to physiological limits ---
        for vital in self.vitals:
            if vital in VITAL_CLAMP_RANGES:
                lo, hi = VITAL_CLAMP_RANGES[vital]
                self.vitals[vital] = float(np.clip(self.vitals[vital], lo, hi))

        for lab in self.labs:
            if lab in LAB_CLAMP_RANGES:
                lo, hi = LAB_CLAMP_RANGES[lab]
                self.labs[lab] = float(np.clip(self.labs[lab], lo, hi))

        # --- Update visible labs if ordered ---
        if labs_ordered:
            self._visible_labs = dict(self.labs)
            self._labs_fresh = True
        else:
            self._labs_fresh = False

        # --- Check for lethal conditions ---
        is_fatal, fatal_reason = self._check_critical()

        return self._make_vitals(), self._make_labs(), self.alerts, is_fatal, fatal_reason

    def _handle_add_drug(self, action: Action, allowed_drugs: List[str]) -> None:
        """Add a new drug infusion."""
        drug = action.drug
        dose = action.dose

        if not drug or dose is None:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="warning",
                message="add_drug requires drug and dose",
                source="action_validation",
            ))
            return

        if drug not in allowed_drugs:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="warning",
                message=f"Drug '{drug}' not allowed for this task",
                source="action_validation",
            ))
            return

        if drug in self.active_drugs:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="info",
                message=f"Drug '{drug}' already active — use titrate to adjust dose",
                source="action_validation",
            ))
            return

        # Validate dose
        if drug in DRUG_DOSE_RANGES:
            min_d, max_d, unit = DRUG_DOSE_RANGES[drug]
            if dose < min_d or dose > max_d:
                self.alerts.append(Alert(
                    severity="warning",
                    message=f"Unsafe dose for {drug}: {dose} (range: {min_d}-{max_d} {unit})",
                    source="dose_validation",
                ))
                # Clamp dose to safe range
                dose = float(np.clip(dose, min_d, max_d))
            self.active_drugs[drug] = DrugInfo(
                drug_name=drug,
                current_dose=dose,
                unit=unit,
                step_started=self.current_step,
            )
        else:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="warning",
                message=f"Unknown drug: {drug}",
                source="action_validation",
            ))

    def _handle_titrate(self, action: Action, allowed_drugs: List[str]) -> None:
        """Titrate (adjust dose of) an existing drug."""
        drug = action.drug
        dose = action.dose

        if not drug or dose is None:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="warning",
                message="titrate requires drug and dose",
                source="action_validation",
            ))
            return

        if drug not in self.active_drugs:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="info",
                message=f"Cannot titrate '{drug}' — not currently active. Use add_drug first.",
                source="action_validation",
            ))
            return

        if drug in DRUG_DOSE_RANGES:
            min_d, max_d, unit = DRUG_DOSE_RANGES[drug]
            if dose < min_d or dose > max_d:
                self.alerts.append(Alert(
                    severity="warning",
                    message=f"Unsafe dose for {drug}: {dose} (range: {min_d}-{max_d} {unit})",
                    source="dose_validation",
                ))
                dose = float(np.clip(dose, min_d, max_d))
            self.active_drugs[drug].current_dose = dose

    def _handle_remove_drug(self, action: Action) -> None:
        """Remove (stop) a drug infusion."""
        drug = action.drug

        if not drug:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="warning",
                message="remove_drug requires drug",
                source="action_validation",
            ))
            return

        if drug not in self.active_drugs:
            self._had_invalid_action = True
            self.alerts.append(Alert(
                severity="info",
                message=f"Cannot remove '{drug}' — not currently active",
                source="action_validation",
            ))
            return

        del self.active_drugs[drug]

    def _apply_interactions(self) -> None:
        """Check and apply drug-drug interaction offsets."""
        active_set = set(self.active_drugs.keys())

        # Check critical interactions
        for drug_pair, (severity, desc, vital, effect) in CRITICAL_INTERACTIONS.items():
            if drug_pair.issubset(active_set):
                if vital in self.vitals:
                    self.vitals[vital] += effect
                elif vital in self.labs:
                    self.labs[vital] += effect
                self.alerts.append(Alert(
                    severity="critical",
                    message=desc,
                    source="drug_interaction",
                ))

        # Check warning interactions
        for drug_pair, (severity, desc, vital, effect) in WARNING_INTERACTIONS.items():
            if drug_pair.issubset(active_set):
                if vital in self.vitals:
                    self.vitals[vital] += effect
                elif vital in self.labs:
                    self.labs[vital] += effect
                self.alerts.append(Alert(
                    severity="warning",
                    message=desc,
                    source="drug_interaction",
                ))

    def _check_critical(self) -> Tuple[bool, str]:
        """Check if any vital is at a critical (lethal) level."""
        for vital, value in self.vitals.items():
            if vital in VITAL_CRITICAL_RANGES:
                lo, hi = VITAL_CRITICAL_RANGES[vital]
                if value <= lo or value >= hi:
                    return True, (
                        f"Critical {vital}: {value:.1f} "
                        f"(safe: {lo}-{hi})"
                    )
        return False, ""

    def get_vitals_in_range(self) -> Dict[str, bool]:
        """Check which vitals are in their safe range."""
        result = {}
        for vital, value in self.vitals.items():
            if vital in VITAL_SAFE_RANGES:
                lo, hi = VITAL_SAFE_RANGES[vital]
                result[vital] = lo <= value <= hi
        return result

    def get_labs_in_range(self) -> Dict[str, bool]:
        """Check which labs are in their safe range (uses ACTUAL labs)."""
        result = {}
        for lab, value in self.labs.items():
            if lab in LAB_SAFE_RANGES:
                lo, hi = LAB_SAFE_RANGES[lab]
                result[lab] = lo <= value <= hi
        return result

    def get_visible_labs_in_range(self) -> Dict[str, bool]:
        """Check which visible labs are in their safe range."""
        result = {}
        for lab, value in self._visible_labs.items():
            if lab in LAB_SAFE_RANGES:
                lo, hi = LAB_SAFE_RANGES[lab]
                result[lab] = lo <= value <= hi
        return result

    def count_vitals_in_range(self) -> int:
        """Count how many vitals are in their safe range."""
        return sum(1 for v in self.get_vitals_in_range().values() if v)

    def count_labs_in_range(self) -> int:
        """Count how many labs are in their safe range (uses ACTUAL labs)."""
        return sum(1 for v in self.get_labs_in_range().values() if v)

    def has_unsafe_dose_alert(self) -> bool:
        """Check if the last action generated an unsafe dose alert."""
        return any(
            a.source == "dose_validation"
            for a in self.alerts
        )

    def has_invalid_action(self) -> bool:
        """Check if the last action was invalid."""
        return self._had_invalid_action

    def has_critical_interaction(self) -> bool:
        """Check if any critical drug interaction is active."""
        return any(
            a.severity == "critical" and a.source == "drug_interaction"
            for a in self.alerts
        )

    def has_warning_interaction(self) -> bool:
        """Check if any warning drug interaction is active."""
        return any(
            a.severity == "warning" and a.source == "drug_interaction"
            for a in self.alerts
        )

    def get_active_drug_list(self) -> List[DrugInfo]:
        """Return list of currently active drug infusions."""
        return list(self.active_drugs.values())

    def _make_vitals(self) -> Vitals:
        """Create a Vitals model from current state."""
        return Vitals(
            map=round(self.vitals["map"], 1),
            hr=round(self.vitals["hr"], 1),
            spo2=round(self.vitals["spo2"], 1),
            rr=round(self.vitals["rr"], 1),
            temp=round(self.vitals["temp"], 2),
        )

    def _make_labs(self) -> Labs:
        """Create a Labs model from current ACTUAL state."""
        return Labs(
            glucose=round(self.labs["glucose"], 1),
            creatinine=round(self.labs["creatinine"], 2),
            potassium=round(self.labs["potassium"], 2),
            lactate=round(self.labs["lactate"], 2),
        )

    def _make_visible_labs(self) -> Labs:
        """Create a Labs model from visible (possibly stale) lab state."""
        return Labs(
            glucose=round(self._visible_labs["glucose"], 1),
            creatinine=round(self._visible_labs["creatinine"], 2),
            potassium=round(self._visible_labs["potassium"], 2),
            lactate=round(self._visible_labs["lactate"], 2),
        )
