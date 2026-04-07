"""
Unit Tests for ICU Drug Titration Environment.

Tests cover:
- Pharmacology constants validation
- Patient simulator determinism
- Environment step/reset/state contracts
- Reward computation
- Grader determinism and bounds
- Action validation
- Drug interaction detection
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pharmacology_constants import (
    ALL_DRUGS,
    CRITICAL_INTERACTIONS,
    DISEASE_PROFILES,
    DRUG_DOSE_RANGES,
    DRUG_EFFECTS,
    LAB_SAFE_RANGES,
    TASK_DEFINITIONS,
    VITAL_SAFE_RANGES,
    WARNING_INTERACTIONS,
)
from models import Action, ActionType, Observation, Reward, State
from patient_simulator import PatientSimulator
from icu_env import ICUDrugTitrationEnv
from grader import grade_easy, grade_medium, grade_hard, grade_trajectory


# =============================================================================
# PHARMACOLOGY CONSTANTS
# =============================================================================

class TestPharmacologyConstants:
    """Tests for pharmacology_constants.py."""

    def test_all_drugs_have_dose_ranges(self):
        """Every drug must have defined dose ranges."""
        for drug in ALL_DRUGS:
            assert drug in DRUG_DOSE_RANGES, f"Missing dose range for {drug}"
            min_d, max_d, unit = DRUG_DOSE_RANGES[drug]
            assert min_d < max_d, f"Invalid range for {drug}: {min_d} >= {max_d}"
            assert len(unit) > 0

    def test_all_drugs_have_effects(self):
        """Every drug must have effect multipliers defined."""
        for drug in ALL_DRUGS:
            assert drug in DRUG_EFFECTS, f"Missing effects for {drug}"
            effects = DRUG_EFFECTS[drug]
            assert "map" in effects
            assert "hr" in effects

    def test_vital_ranges_valid(self):
        """Vital safe ranges must be valid intervals."""
        for vital, (lo, hi) in VITAL_SAFE_RANGES.items():
            assert lo < hi, f"Invalid vital range for {vital}"

    def test_lab_ranges_valid(self):
        """Lab safe ranges must be valid intervals."""
        for lab, (lo, hi) in LAB_SAFE_RANGES.items():
            assert lo < hi, f"Invalid lab range for {lab}"

    def test_disease_profiles_valid(self):
        """All disease profiles must have required fields."""
        assert len(DISEASE_PROFILES) == 3
        for name, profile in DISEASE_PROFILES.items():
            assert len(profile.baseline_vitals) == 5
            assert len(profile.baseline_labs) == 4
            assert len(profile.deterioration_per_hour) > 0
            assert len(profile.allowed_drugs) > 0

    def test_task_definitions_valid(self):
        """All task definitions must be properly configured."""
        assert set(TASK_DEFINITIONS.keys()) == {"easy", "medium", "hard"}
        for task_id, task in TASK_DEFINITIONS.items():
            assert task["horizon"] > 0
            assert task["disease"] in DISEASE_PROFILES
            assert len(task["allowed_drugs"]) > 0

    def test_interactions_use_valid_drugs(self):
        """All drug interactions must reference valid drugs."""
        for pair in CRITICAL_INTERACTIONS:
            for drug in pair:
                assert drug in ALL_DRUGS, f"Unknown drug in critical interaction: {drug}"
        for pair in WARNING_INTERACTIONS:
            for drug in pair:
                assert drug in ALL_DRUGS, f"Unknown drug in warning interaction: {drug}"


# =============================================================================
# PATIENT SIMULATOR
# =============================================================================

class TestPatientSimulator:
    """Tests for patient_simulator.py."""

    def test_reset_returns_vitals_and_labs(self):
        """Reset must return valid Vitals and Labs."""
        sim = PatientSimulator("vasopressor_shock", seed=42)
        vitals, labs = sim.reset()
        assert vitals.map == 52.0
        assert vitals.hr == 110.0
        assert labs.lactate == 3.5

    def test_determinism_same_seed(self):
        """Same seed must produce identical trajectories."""
        def run_trajectory(seed):
            sim = PatientSimulator("vasopressor_shock", seed=seed)
            sim.reset()
            results = []
            for i in range(5):
                action = Action(action_type=ActionType.HOLD)
                vitals, labs, alerts, fatal, reason = sim.apply_action(
                    action, ["norepinephrine"], i + 1
                )
                results.append((vitals.map, vitals.hr, labs.lactate))
            return results

        r1 = run_trajectory(42)
        r2 = run_trajectory(42)
        assert r1 == r2, "Trajectories with same seed must be identical"

    def test_different_seeds_differ(self):
        """Different seeds should produce different trajectories."""
        def run_trajectory(seed):
            sim = PatientSimulator("vasopressor_shock", seed=seed)
            sim.reset()
            action = Action(action_type=ActionType.HOLD)
            vitals, _, _, _, _ = sim.apply_action(action, ["norepinephrine"], 1)
            return vitals.map

        v1 = run_trajectory(42)
        v2 = run_trajectory(99)
        assert v1 != v2

    def test_norepinephrine_increases_map(self):
        """Norepinephrine should increase MAP."""
        sim = PatientSimulator("vasopressor_shock", seed=42)
        sim.reset()
        initial_map = sim.vitals["map"]

        action = Action(
            action_type=ActionType.ADD_DRUG,
            drug="norepinephrine",
            dose=0.2,
        )
        vitals, _, _, _, _ = sim.apply_action(action, ["norepinephrine"], 1)

        # MAP should increase (drug effect should outweigh deterioration)
        assert vitals.map > initial_map - 5, "Norepinephrine should increase MAP"

    def test_invalid_disease_raises(self):
        """Invalid disease name should raise ValueError."""
        with pytest.raises(ValueError):
            PatientSimulator("nonexistent_disease")

    def test_drug_dose_validation(self):
        """Out-of-range dose should generate a warning alert."""
        sim = PatientSimulator("vasopressor_shock", seed=42)
        sim.reset()

        action = Action(
            action_type=ActionType.ADD_DRUG,
            drug="norepinephrine",
            dose=999.0,  # Way above max
        )
        _, _, alerts, _, _ = sim.apply_action(action, ["norepinephrine"], 1)
        assert any(a.source == "dose_validation" for a in alerts)

    def test_drug_interaction_detection(self):
        """Critical interaction should be detected."""
        sim = PatientSimulator("ventilated_sedation", seed=42)
        sim.reset()
        allowed = ["propofol", "fentanyl", "norepinephrine"]

        # Add propofol
        action1 = Action(action_type=ActionType.ADD_DRUG, drug="propofol", dose=30.0)
        sim.apply_action(action1, allowed, 1)

        # Add fentanyl (should trigger critical interaction)
        action2 = Action(action_type=ActionType.ADD_DRUG, drug="fentanyl", dose=50.0)
        _, _, alerts, _, _ = sim.apply_action(action2, allowed, 2)

        assert sim.has_critical_interaction(), "propofol + fentanyl should trigger critical interaction"

    def test_remove_drug(self):
        """Removing a drug should clear it from active drugs."""
        sim = PatientSimulator("vasopressor_shock", seed=42)
        sim.reset()

        add = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1)
        sim.apply_action(add, ["norepinephrine"], 1)
        assert "norepinephrine" in sim.active_drugs

        remove = Action(action_type=ActionType.REMOVE_DRUG, drug="norepinephrine")
        sim.apply_action(remove, ["norepinephrine"], 2)
        assert "norepinephrine" not in sim.active_drugs


# =============================================================================
# ICU ENVIRONMENT
# =============================================================================

class TestICUEnvironment:
    """Tests for icu_env.py."""

    def test_reset_returns_observation(self):
        """Reset must return a valid Observation."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        assert isinstance(obs, Observation)
        assert obs.task_id == "easy"
        assert obs.current_step == 0
        assert sid is not None

    def test_step_returns_correct_types(self):
        """Step must return (Observation, Reward, bool, dict)."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.HOLD)
        obs, reward, done, info = env.step(action, sid)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_full_state(self):
        """State must return complete episode history."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.HOLD)
        env.step(action, sid)
        state = env.state(sid)
        assert isinstance(state, State)
        assert state.current_step == 1
        assert len(state.history) == 1

    def test_episode_ends_at_horizon(self):
        """Episode must end when horizon is reached."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.HOLD)
        for i in range(12):
            obs, reward, done, info = env.step(action, sid)
        assert done
        assert "horizon_reached" in info.get("done_reason", "")

    def test_flag_physician_ends_episode(self):
        """Flag physician action must end the episode."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.FLAG_PHYSICIAN)
        obs, reward, done, info = env.step(action, sid)
        assert done
        assert reward.value < 0  # Penalty

    def test_step_on_done_raises(self):
        """Stepping on a done episode must raise ValueError."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.FLAG_PHYSICIAN)
        env.step(action, sid)
        with pytest.raises(ValueError):
            env.step(Action(action_type=ActionType.HOLD), sid)

    def test_invalid_session_raises(self):
        """Invalid session ID must raise ValueError."""
        env = ICUDrugTitrationEnv()
        with pytest.raises(ValueError):
            env.step(Action(action_type=ActionType.HOLD), "nonexistent")

    def test_multiple_sessions(self):
        """Multiple concurrent sessions must be independent."""
        env = ICUDrugTitrationEnv()
        obs1, sid1 = env.reset(task_id="easy", seed=42)
        obs2, sid2 = env.reset(task_id="medium", seed=42)
        assert sid1 != sid2
        assert obs1.task_id == "easy"
        assert obs2.task_id == "medium"

    def test_reward_is_dense(self):
        """Every step must produce a non-None reward."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)
        action = Action(action_type=ActionType.HOLD)
        for _ in range(3):
            obs, reward, done, info = env.step(action, sid)
            assert reward.value is not None
            assert len(reward.breakdown) > 0


# =============================================================================
# GRADER
# =============================================================================

class TestGrader:
    """Tests for grader.py."""

    def _run_episode(self, task_id, seed=42):
        """Helper to run a complete episode."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id=task_id, seed=seed)
        action = Action(
            action_type=ActionType.ADD_DRUG,
            drug="norepinephrine",
            dose=0.1,
        )
        env.step(action, sid)
        while True:
            obs, reward, done, info = env.step(
                Action(action_type=ActionType.HOLD), sid
            )
            if done:
                break
        return env.state(sid)

    def test_easy_score_in_bounds(self):
        """Easy grader score must be in [0.0, 1.0]."""
        state = self._run_episode("easy")
        score, breakdown = grade_easy(state)
        assert 0.0 <= score <= 1.0

    def test_medium_score_in_bounds(self):
        """Medium grader score must be in [0.0, 1.0]."""
        state = self._run_episode("medium")
        score, breakdown = grade_medium(state)
        assert 0.0 <= score <= 1.0

    def test_hard_score_in_bounds(self):
        """Hard grader score must be in [0.0, 1.0]."""
        state = self._run_episode("hard")
        score, breakdown = grade_hard(state)
        assert 0.0 <= score <= 1.0

    def test_grader_determinism(self):
        """Same trajectory must always yield same score."""
        state1 = self._run_episode("easy", seed=42)
        state2 = self._run_episode("easy", seed=42)
        score1, _ = grade_easy(state1)
        score2, _ = grade_easy(state2)
        assert score1 == score2, "Grader must be deterministic"

    def test_grade_trajectory_dispatches(self):
        """grade_trajectory must route to correct grader."""
        for task_id in ["easy", "medium", "hard"]:
            state = self._run_episode(task_id)
            score, breakdown = grade_trajectory(state)
            assert 0.0 <= score <= 1.0

    def test_grade_empty_history(self):
        """Grading with no steps should return 0.0."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy")
        # Force done without steps
        state = env.state(sid)
        state.done = True
        state.done_reason = "test"
        state.history = []
        score, breakdown = grade_easy(state)
        assert score == 0.0


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_easy_episode(self):
        """Run a complete easy episode end-to-end."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="easy", seed=42)

        # Add drug
        action = Action(
            action_type=ActionType.ADD_DRUG,
            drug="norepinephrine",
            dose=0.1,
        )
        obs, reward, done, info = env.step(action, sid)
        assert not done

        # Run to completion
        while not done:
            obs, reward, done, info = env.step(
                Action(action_type=ActionType.HOLD), sid
            )

        # Grade
        state = env.state(sid)
        score, breakdown = grade_trajectory(state)
        assert 0.0 <= score <= 1.0
        assert state.current_step == 12
        assert state.done

    def test_full_hard_episode(self):
        """Run a complete hard episode end-to-end."""
        env = ICUDrugTitrationEnv()
        obs, sid = env.reset(task_id="hard", seed=42)

        # Add multiple drugs
        env.step(Action(
            action_type=ActionType.ADD_DRUG,
            drug="norepinephrine", dose=0.15
        ), sid)
        env.step(Action(
            action_type=ActionType.ADD_DRUG,
            drug="insulin", dose=5.0
        ), sid)

        # Run to completion
        done = False
        while not done:
            _, _, done, _ = env.step(Action(action_type=ActionType.HOLD), sid)

        state = env.state(sid)
        score, _ = grade_trajectory(state)
        assert 0.0 <= score <= 1.0
        assert len(state.history) == state.current_step
        assert state.done


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
