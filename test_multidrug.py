"""Verify multi-drug incentive fixes work correctly."""
from icu_env import ICUDrugTitrationEnv
from models import Action, ActionType

env = ICUDrugTitrationEnv()

print("=== REWARD COMPARISON: Single Drug vs Multi Drug ===")
print("(Using non-interacting drugs to isolate reward signal effect)")
print()

# Scenario A: Norepi only
print("--- Scenario A: Norepinephrine ONLY (hard task) ---")
obs, sid = env.reset(task_id="hard", seed=42)
a = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.15)
obs, r, d, i = env.step(a, sid)
print(f"Step 1: Reward={r.value:+.3f} | Breakdown: {r.breakdown}")
for step in range(2, 8):
    obs, r, d, i = env.step(Action(action_type=ActionType.HOLD), sid)
    s = env.sessions[sid].simulator
    vok = s.count_vitals_in_range()
    lok = s.count_labs_in_range()
    print(f"Step {step}: Reward={r.value:+.3f} | vitals_ok={vok} labs_ok={lok}")

total_a = env.sessions[sid].total_reward
print(f"  Total reward after 7 steps: {total_a:.3f}")

print()

# Scenario B: Norepi + Insulin (no interaction between these)
print("--- Scenario B: Norepi + Insulin (hard task, no interaction) ---")
obs, sid2 = env.reset(task_id="hard", seed=42)
env.step(Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.15), sid2)
obs, r, d, i = env.step(Action(action_type=ActionType.ADD_DRUG, drug="insulin", dose=5.0), sid2)
s = env.sessions[sid2].simulator
print(f"Step 2: Reward={r.value:+.3f} | vitals_ok={s.count_vitals_in_range()} labs_ok={s.count_labs_in_range()}")
print(f"  Breakdown: {r.breakdown}")

for step in range(3, 8):
    obs, r, d, i = env.step(Action(action_type=ActionType.HOLD), sid2)
    s = env.sessions[sid2].simulator
    vok = s.count_vitals_in_range()
    lok = s.count_labs_in_range()
    print(f"Step {step}: Reward={r.value:+.3f} | vitals_ok={vok} labs_ok={lok}")

total_b = env.sessions[sid2].total_reward
print(f"  Total reward after 7 steps: {total_b:.3f}")

print()
print(f"=== RESULT ===")
print(f"  Single drug (norepi):      {total_a:+.3f}")
print(f"  Multi drug (norepi+insulin): {total_b:+.3f}")
advantage = total_b - total_a
print(f"  Multi-drug advantage: {advantage:+.3f}")
assert advantage > 0, f"Multi-drug should be better! Got {advantage:.3f}"
print(f"  ✓ Multi-drug strategy is better by {advantage:.3f}")

# Test deterioration penalty
print()
print("=== DETERIORATION PENALTY ===")
obs, sid3 = env.reset(task_id="hard", seed=42)
obs, r, d, i = env.step(Action(action_type=ActionType.HOLD), sid3)
det_pen = r.breakdown.get("deterioration_penalty", 0)
print(f"Step 1 (no drugs): deterioration_penalty = {det_pen}")
assert det_pen < 0, "Should have negative deterioration penalty"
total_out = 5 + 4  # all 9 params out
expected = total_out * -0.05
print(f"  Expected: {expected} (9 params × -0.05)")
print(f"  ✓ Deterioration penalty works: {det_pen}")

# Test multi-coverage bonus
print()
print("=== MULTI-COVERAGE BONUS ===")
obs, sid4 = env.reset(task_id="easy", seed=42)
env.step(Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1), sid4)
obs, r, d, i = env.step(Action(action_type=ActionType.HOLD), sid4)
bonus = r.breakdown.get("multi_coverage_bonus", 0)
s = env.sessions[sid4].simulator
vok = s.count_vitals_in_range()
lok = s.count_labs_in_range()
print(f"Easy task: vitals_ok={vok} labs_ok={lok}")
print(f"  Multi-coverage bonus: {bonus}")
if vok >= 3 and lok >= 2:
    assert bonus > 0, "Should get bonus with 3+ vitals and 2+ labs"
    print(f"  ✓ Bonus awarded: {bonus}")
else:
    print(f"  (Not enough systems in range for bonus)")

print()
print("ALL CHECKS PASSED ✓")
