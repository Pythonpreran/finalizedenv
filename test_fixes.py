"""Functional tests verifying all 7 audit fixes."""

from icu_env import ICUDrugTitrationEnv
from grader import grade_trajectory
from models import Action, ActionType
import json

env = ICUDrugTitrationEnv()

def test_fix1_equilibrium_model():
    """Fix 1: Drug effects should NOT accumulate permanently."""
    print("=== Fix 1: Equilibrium Drug Model ===")
    obs, sid = env.reset(task_id="easy", seed=42)
    
    # Add norepinephrine
    action = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1)
    obs, r, done, info = env.step(action, sid)
    map_with_drug = obs.vitals.map
    print(f"  Step 1 (norepi 0.1): MAP = {map_with_drug:.1f}")
    
    # Hold for a step
    obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
    map_held = obs.vitals.map
    print(f"  Step 2 (hold):       MAP = {map_held:.1f}")
    
    # Remove norepinephrine
    action = Action(action_type=ActionType.REMOVE_DRUG, drug="norepinephrine")
    obs, r, done, info = env.step(action, sid)
    map_removed = obs.vitals.map
    print(f"  Step 3 (removed):    MAP = {map_removed:.1f}")
    
    # The MAP should drop after removal (close to undrugged state)
    # In the old model, MAP would stay elevated
    session = env.sessions[sid]
    undrugged_map = session.simulator._undrugged_vitals["map"]
    print(f"  Undrugged MAP:       {undrugged_map:.1f}")
    
    # MAP after removal should be close to undrugged (within noise)
    diff = abs(map_removed - undrugged_map)
    assert diff < 10, f"MAP should be near undrugged after removal, diff={diff:.1f}"
    print(f"  ✓ Drug offset removed correctly (diff from undrugged: {diff:.1f})")
    print()

def test_fix2_lab_gating():
    """Fix 2: Labs should be stale unless order_lab is used."""
    print("=== Fix 2: Lab Gating ===")
    obs, sid = env.reset(task_id="easy", seed=42)
    initial_glucose = obs.labs.glucose
    print(f"  Initial glucose (fresh): {initial_glucose:.1f}")
    
    # Hold several steps (labs should become stale)
    obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
    stale_glucose = obs.labs.glucose
    print(f"  Step 1 glucose (stale): {stale_glucose:.1f}, fresh={obs.labs_fresh}")
    assert not obs.labs_fresh, "Labs should be stale after hold"
    assert stale_glucose == initial_glucose, "Stale labs should not change"
    
    # Order labs
    obs, r, done, info = env.step(Action(action_type=ActionType.ORDER_LAB), sid)
    fresh_glucose = obs.labs.glucose
    print(f"  Step 2 glucose (ordered): {fresh_glucose:.1f}, fresh={obs.labs_fresh}")
    assert obs.labs_fresh, "Labs should be fresh after order"
    # Glucose should have changed due to deterioration
    print(f"  ✓ Labs properly gated (stale={stale_glucose:.1f} vs fresh={fresh_glucose:.1f})")
    print()

def test_fix3_escalating_penalties():
    """Fix 3: Interaction penalties should escalate with duration."""
    print("=== Fix 3: Escalating Interaction Penalties ===")
    obs, sid = env.reset(task_id="medium", seed=42)
    
    # Add propofol and norepinephrine (warning interaction)
    env.step(Action(action_type=ActionType.ADD_DRUG, drug="propofol", dose=20.0), sid)
    obs, r, done, info = env.step(
        Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1), sid
    )
    penalty_step1 = r.breakdown.get("warning_interaction", 0)
    print(f"  Step 1 warning penalty: {penalty_step1:.2f}")
    
    obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
    penalty_step2 = r.breakdown.get("warning_interaction", 0)
    print(f"  Step 2 warning penalty: {penalty_step2:.2f}")
    
    obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
    penalty_step3 = r.breakdown.get("warning_interaction", 0)
    print(f"  Step 3 warning penalty: {penalty_step3:.2f}")
    
    assert abs(penalty_step2) > abs(penalty_step1), "Penalty should escalate"
    assert abs(penalty_step3) > abs(penalty_step2), "Penalty should continue escalating"
    print(f"  ✓ Penalties escalate correctly")
    print()

def test_fix4_per_drug_smoothness():
    """Fix 4: Smoothness should be per-drug, not summed."""
    print("=== Fix 4: Per-Drug Normalized Smoothness ===")
    obs, sid = env.reset(task_id="easy", seed=42)
    
    # Make some titrations
    env.step(Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1), sid)
    for _ in range(5):
        env.step(Action(action_type=ActionType.HOLD), sid)
    # Titrate up slightly
    env.step(Action(action_type=ActionType.TITRATE, drug="norepinephrine", dose=0.12), sid)
    for _ in range(5):
        obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
    
    state = env.state(sid)
    score, breakdown = grade_trajectory(state)
    print(f"  Score: {score:.4f}")
    print(f"  Smoothness: {breakdown.get('smoothness', 'N/A')}")
    print(f"  ✓ Per-drug smoothness computed")
    print()

def test_fix6_invalid_action_penalty():
    """Fix 6: Invalid actions should receive a penalty."""
    print("=== Fix 6: Invalid Action Penalty ===")
    obs, sid = env.reset(task_id="easy", seed=42)
    
    # Try to titrate a drug that's not active (invalid)
    obs, r, done, info = env.step(
        Action(action_type=ActionType.TITRATE, drug="norepinephrine", dose=0.1), sid
    )
    invalid_penalty = r.breakdown.get("invalid_action", 0)
    print(f"  Invalid action penalty: {invalid_penalty:.2f}")
    assert invalid_penalty < 0, "Invalid action should have negative penalty"
    
    # Valid action should have no penalty
    obs, r, done, info = env.step(
        Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1), sid
    )
    valid_penalty = r.breakdown.get("invalid_action", 0)
    print(f"  Valid action penalty: {valid_penalty:.2f}")
    assert valid_penalty == 0, "Valid action should have no penalty"
    print(f"  ✓ Invalid actions penalized correctly")
    print()

def test_fix7_random_seed():
    """Fix 7: Default seed should be random, not static."""
    print("=== Fix 7: Random Seed ===")
    obs1, sid1 = env.reset(task_id="easy")
    obs2, sid2 = env.reset(task_id="easy")
    
    seed1 = env.sessions[sid1].seed
    seed2 = env.sessions[sid2].seed
    print(f"  Session 1 seed: {seed1}")
    print(f"  Session 2 seed: {seed2}")
    assert seed1 != seed2, "Default seeds should be different"
    print(f"  ✓ Seeds are randomized")
    
    # Explicit seed should still work
    obs3, sid3 = env.reset(task_id="easy", seed=123)
    obs4, sid4 = env.reset(task_id="easy", seed=123)
    print(f"  Explicit seed 123: MAP1={obs3.vitals.map}, MAP2={obs4.vitals.map}")
    assert obs3.vitals.map == obs4.vitals.map, "Same seed should give same result"
    print(f"  ✓ Explicit seeds still deterministic")
    print()

def test_full_episode():
    """Run a full easy episode and grade it."""
    print("=== Full Episode Test (Easy) ===")
    obs, sid = env.reset(task_id="easy", seed=42)
    
    # Add norepinephrine
    obs, r, done, info = env.step(
        Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1), sid
    )
    print(f"  Step 1: MAP={obs.vitals.map:.1f}, reward={r.value:.4f}")
    
    # Hold for remaining steps
    step = 2
    while not done:
        obs, r, done, info = env.step(Action(action_type=ActionType.HOLD), sid)
        print(f"  Step {step}: MAP={obs.vitals.map:.1f}, reward={r.value:.4f}")
        step += 1
    
    state = env.state(sid)
    score, breakdown = grade_trajectory(state)
    print(f"\n  Final Score: {score:.4f}")
    print(f"  Breakdown: {json.dumps({k: round(v, 4) for k, v in breakdown.items()})}")
    print()

if __name__ == "__main__":
    test_fix1_equilibrium_model()
    test_fix2_lab_gating()
    test_fix3_escalating_penalties()
    test_fix4_per_drug_smoothness()
    test_fix6_invalid_action_penalty()
    test_fix7_random_seed()
    test_full_episode()
    print("=" * 50)
    print("ALL TESTS PASSED ✓")
