"""
ICU Drug Titration Environment.

OpenEnv-compliant RL environment for ICU clinical pharmacology simulation.
Implements step(), reset(), state() interface with dense reward shaping.

Fixes applied:
- Random seed by default (no memorization)
- Invalid action penalty
- Escalating interaction penalties (duration-based)
- Lab visibility gated behind order_lab
- Actual labs tracked in StepRecord for grading
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from pharmacology_constants import (
    REWARD_CRITICAL_INTERACTION,
    REWARD_DETERIORATION_PENALTY,
    REWARD_FLAG_PHYSICIAN,
    REWARD_INVALID_ACTION,
    REWARD_LAB_IN_RANGE,
    REWARD_MULTI_COVERAGE_BONUS,
    REWARD_PATIENT_DEATH,
    REWARD_TERMINAL_BONUS,
    REWARD_UNSAFE_DOSE,
    REWARD_VITAL_IN_RANGE,
    REWARD_WARNING_INTERACTION,
    TASK_DEFINITIONS,
)
from models import (
    Action,
    ActionType,
    Alert,
    Labs,
    Observation,
    Reward,
    State,
    StepRecord,
    Vitals,
)
from patient_simulator import PatientSimulator


class ICUDrugTitrationEnv:
    """
    OpenEnv-compliant ICU Drug Titration Environment.

    An AI agent acts as an ICU clinical pharmacist, making drug titration
    decisions over a simulated 24-hour patient episode. Each step represents
    one simulated hour.

    The environment supports three difficulty levels:
    - easy: Single drug MAP control (12 steps)
    - medium: Multi-drug ventilated sedation (20 steps)
    - hard: Septic shock with renal failure (24 steps)
    """

    def __init__(self):
        """Initialize the environment manager."""
        self.sessions: Dict[str, _Session] = {}

    def reset(
        self,
        task_id: str = "easy",
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Observation, str]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Task difficulty level (easy, medium, hard).
            session_id: Optional session ID for multi-session support.
            seed: Random seed for reproducibility. If None, a random seed is generated.

        Returns:
            Tuple of (initial_observation, session_id).
        """
        if task_id not in TASK_DEFINITIONS:
            raise ValueError(
                f"Unknown task: {task_id}. Available: {list(TASK_DEFINITIONS.keys())}"
            )

        if session_id is None:
            session_id = str(uuid.uuid4())

        task = TASK_DEFINITIONS[task_id]
        # Fix #7: Random seed by default instead of static seed=42
        if seed is not None:
            _seed = seed
        else:
            _seed = int(np.random.randint(0, 2**31))

        session = _Session(
            task_id=task_id,
            task=task,
            session_id=session_id,
            seed=_seed,
        )
        self.sessions[session_id] = session

        return session.get_observation(), session_id

    def step(
        self,
        action: Action,
        session_id: str,
    ) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: The clinical action to execute.
            session_id: Session identifier.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")

        session = self.sessions[session_id]

        if session.done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        return session.step(action)

    def state(self, session_id: str) -> State:
        """
        Get the full environment state for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Complete State object with episode history.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")

        return self.sessions[session_id].get_state()

    def get_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.sessions.keys())


class _Session:
    """
    Internal session managing a single episode.

    Encapsulates the patient simulator, step tracking, and reward computation
    for one episode of the environment.
    """

    def __init__(
        self,
        task_id: str,
        task: Dict,
        session_id: str,
        seed: int,
    ):
        self.task_id = task_id
        self.task = task
        self.session_id = session_id
        self.episode_id = str(uuid.uuid4())
        self.seed = seed

        # Initialize simulator
        self.simulator = PatientSimulator(
            disease_name=task["disease"],
            seed=seed,
        )
        self.simulator.reset()

        # Episode state
        self.current_step = 0
        self.max_steps = task["horizon"]
        self.done = False
        self.done_reason: Optional[str] = None
        self.total_reward = 0.0
        self.history: List[StepRecord] = []

        # Fix #3: Track consecutive interaction steps for escalation
        self._consecutive_critical_interaction_steps = 0
        self._consecutive_warning_interaction_steps = 0

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Execute one step and return the result."""
        self.current_step += 1
        terminated = False
        truncated = False

        # Handle flag_physician immediately
        if action.action_type == ActionType.FLAG_PHYSICIAN:
            self.done = True
            self.done_reason = "flag_physician"
            terminated = True
            reward = self._compute_reward(
                flag_physician=True,
                is_fatal=False,
            )
            obs = self.get_observation()
            self._record_step(action, obs, reward)
            return obs, reward, True, {
                "done_reason": "flag_physician",
                "terminated": True,
                "truncated": False,
            }

        # Apply action to simulator
        vitals, labs, alerts, is_fatal, fatal_reason = self.simulator.apply_action(
            action=action,
            allowed_drugs=self.task["allowed_drugs"],
            step=self.current_step,
        )

        # Update interaction duration tracking
        if self.simulator.has_critical_interaction():
            self._consecutive_critical_interaction_steps += 1
        else:
            self._consecutive_critical_interaction_steps = 0

        if self.simulator.has_warning_interaction():
            self._consecutive_warning_interaction_steps += 1
        else:
            self._consecutive_warning_interaction_steps = 0

        # Check for patient death
        if is_fatal:
            self.done = True
            self.done_reason = f"patient_death: {fatal_reason}"
            terminated = True

        # Check for horizon
        if self.current_step >= self.max_steps:
            self.done = True
            if self.done_reason is None:
                self.done_reason = "horizon_reached"
            truncated = not terminated  # Only truncated if not already terminated

        # Compute reward
        reward = self._compute_reward(
            flag_physician=False,
            is_fatal=is_fatal,
        )

        obs = self.get_observation()
        self._record_step(action, obs, reward)

        info = {
            "terminated": terminated,
            "truncated": truncated,
        }
        if self.done:
            info["done_reason"] = self.done_reason

        return obs, reward, self.done, info

    def _compute_reward(
        self,
        flag_physician: bool,
        is_fatal: bool,
    ) -> Reward:
        """
        Compute dense reward for the current step.

        Reward components:
        - +0.1 per vital in safe range
        - +0.1 per lab in safe range (raised from 0.05 to incentivize lab management)
        - +0.15 multi-system coverage bonus (≥3 vitals + ≥2 labs in range)
        - -0.05 per vital/lab OUT of range (deterioration pressure)
        - -1.0 × duration for critical drug interaction (escalating)
        - -0.3 × duration for warning drug interaction (escalating)
        - -0.5 for unsafe dose
        - -0.2 for invalid action
        - +2.0 terminal bonus if all vitals+labs stable
        - -1.5 for flagging physician (giving up)
        - -5.0 for patient death
        """
        breakdown = {}
        total = 0.0

        if flag_physician:
            breakdown["flag_physician"] = REWARD_FLAG_PHYSICIAN
            total += REWARD_FLAG_PHYSICIAN
        elif is_fatal:
            breakdown["patient_death"] = REWARD_PATIENT_DEATH
            total += REWARD_PATIENT_DEATH
        else:
            # Vitals in range reward
            vitals_in_range = self.simulator.count_vitals_in_range()
            vital_reward = vitals_in_range * REWARD_VITAL_IN_RANGE
            breakdown["vitals_in_range"] = vital_reward
            total += vital_reward

            # Labs in range reward (now equal to vitals — labs matter!)
            labs_in_range = self.simulator.count_labs_in_range()
            lab_reward = labs_in_range * REWARD_LAB_IN_RANGE
            breakdown["labs_in_range"] = lab_reward
            total += lab_reward

            # Multi-system coverage bonus: reward when multiple systems are stable
            if vitals_in_range >= 3 and labs_in_range >= 2:
                breakdown["multi_coverage_bonus"] = REWARD_MULTI_COVERAGE_BONUS
                total += REWARD_MULTI_COVERAGE_BONUS

            # Deterioration penalty: per parameter OUT of range
            vitals_out = 5 - vitals_in_range
            labs_out = 4 - labs_in_range
            total_out = vitals_out + labs_out
            if total_out > 0:
                det_penalty = total_out * REWARD_DETERIORATION_PENALTY
                breakdown["deterioration_penalty"] = det_penalty
                total += det_penalty

            # Escalating interaction penalties
            if self.simulator.has_critical_interaction():
                duration_factor = self._consecutive_critical_interaction_steps
                penalty = REWARD_CRITICAL_INTERACTION * duration_factor
                breakdown["critical_interaction"] = penalty
                total += penalty

            if self.simulator.has_warning_interaction():
                duration_factor = self._consecutive_warning_interaction_steps
                penalty = REWARD_WARNING_INTERACTION * duration_factor
                breakdown["warning_interaction"] = penalty
                total += penalty

            # Unsafe dose penalty
            if self.simulator.has_unsafe_dose_alert():
                breakdown["unsafe_dose"] = REWARD_UNSAFE_DOSE
                total += REWARD_UNSAFE_DOSE

            # Invalid action penalty
            if self.simulator.has_invalid_action():
                breakdown["invalid_action"] = REWARD_INVALID_ACTION
                total += REWARD_INVALID_ACTION

            # Terminal bonus
            if self.done and self.done_reason == "horizon_reached":
                all_vitals_safe = vitals_in_range == 5
                all_labs_safe = labs_in_range == 4
                if all_vitals_safe and all_labs_safe:
                    breakdown["terminal_bonus"] = REWARD_TERMINAL_BONUS
                    total += REWARD_TERMINAL_BONUS

        total = round(total, 4)
        self.total_reward += total

        return Reward(value=total, breakdown=breakdown)

    def get_observation(self) -> Observation:
        """Build the current observation (agent-visible, with stale lab gating)."""
        return Observation(
            vitals=self.simulator._make_vitals(),
            labs=self.simulator._make_visible_labs(),        # Fix #2: visible (gated) labs
            labs_fresh=self.simulator._labs_fresh,
            active_drugs=self.simulator.get_active_drug_list(),
            current_step=self.current_step,
            max_steps=self.max_steps,
            disease=self.task["disease"],
            task_id=self.task_id,
            alerts=list(self.simulator.alerts),
            vitals_in_range=self.simulator.get_vitals_in_range(),
            labs_in_range=self.simulator.get_visible_labs_in_range(),  # Fix #2: visible labs
        )

    def get_state(self) -> State:
        """Build the full environment state."""
        return State(
            episode_id=self.episode_id,
            session_id=self.session_id,
            task_id=self.task_id,
            disease=self.task["disease"],
            current_step=self.current_step,
            max_steps=self.max_steps,
            done=self.done,
            done_reason=self.done_reason,
            total_reward=round(self.total_reward, 4),
            history=list(self.history),
            current_vitals=self.simulator._make_vitals(),
            current_labs=self.simulator._make_labs(),
            active_drugs=self.simulator.get_active_drug_list(),
        )

    def _record_step(self, action: Action, obs: Observation, reward: Reward) -> None:
        """Record a step in the episode history, including actual labs for grading."""
        self.history.append(StepRecord(
            step=self.current_step,
            action=action,
            observation=obs,
            reward=reward,
            actual_labs=self.simulator._make_labs(),
            actual_labs_in_range=self.simulator.get_labs_in_range(),
        ))
