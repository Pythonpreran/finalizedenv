"""
Baseline Script for ICU Drug Titration Environment.

Uses the OpenAI API to run an LLM agent through all 3 tasks
and outputs reproducible scores. Evaluation is deterministic.

Usage:
    export OPENAI_API_KEY=your_api_key
    python baseline.py [--server-url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

# Optional OpenAI import
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are an expert ICU clinical pharmacist AI agent.
You are managing drug titration in a simulated ICU patient.

Your goal is to keep all patient vital signs and lab values within safe ranges
using the available drugs.

Safe vital ranges:
- MAP: 65-90 mmHg
- HR: 60-100 bpm
- SpO2: 94-100%
- RR: 12-20 breaths/min
- Temp: 36.5-37.5°C

Safe lab ranges:
- Glucose: 70-180 mg/dL
- Creatinine: 0.6-1.2 mg/dL
- Potassium: 3.5-5.0 mEq/L
- Lactate: 0.5-2.0 mmol/L

You must respond with a JSON action. Available actions:
- {"action_type": "add_drug", "drug": "<name>", "dose": <number>}
- {"action_type": "titrate", "drug": "<name>", "dose": <number>}
- {"action_type": "remove_drug", "drug": "<name>"}
- {"action_type": "hold"}
- {"action_type": "order_lab"}
- {"action_type": "flag_physician"}

Be careful with drug interactions:
- propofol + fentanyl → respiratory depression (CRITICAL)
- vasopressin + norepinephrine → peripheral ischemia (CRITICAL)

Respond ONLY with the JSON action, no explanation."""


def run_heuristic_baseline(server_url: str) -> dict:
    """
    Run a simple heuristic (non-LLM) baseline for deterministic scoring.

    This provides a reproducible baseline without requiring an API key.
    """
    client = httpx.Client(base_url=server_url, timeout=30.0)
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Running heuristic baseline: {task_id}")
        print(f"{'='*50}")

        # Reset
        res = client.post("/reset", json={"task_id": task_id, "seed": 42})
        data = res.json()
        session_id = data["session_id"]
        obs = data["observation"]

        total_reward = 0
        steps = 0

        # Heuristic strategy per task
        if task_id == "easy":
            # Add norepinephrine and titrate based on MAP
            action = {"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1}
            res = client.post(f"/step?session_id={session_id}", json=action)
            step_data = res.json()
            total_reward += step_data["reward"]["value"]
            obs = step_data["observation"]
            steps += 1

            while not step_data["done"]:
                map_val = obs["vitals"]["map"]
                current_dose = 0.1
                for d in obs["active_drugs"]:
                    if d["drug_name"] == "norepinephrine":
                        current_dose = d["current_dose"]

                if map_val < 65:
                    new_dose = min(0.5, current_dose + 0.05)
                    action = {"action_type": "titrate", "drug": "norepinephrine", "dose": new_dose}
                elif map_val > 90:
                    new_dose = max(0.01, current_dose - 0.05)
                    action = {"action_type": "titrate", "drug": "norepinephrine", "dose": new_dose}
                else:
                    action = {"action_type": "hold"}

                res = client.post(f"/step?session_id={session_id}", json=action)
                step_data = res.json()
                total_reward += step_data["reward"]["value"]
                obs = step_data["observation"]
                steps += 1

        elif task_id == "medium":
            # Add norepinephrine, then propofol (avoiding interaction by keeping doses moderate)
            actions_init = [
                {"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.05},
                {"action_type": "add_drug", "drug": "propofol", "dose": 25.0},
            ]
            for action in actions_init:
                res = client.post(f"/step?session_id={session_id}", json=action)
                step_data = res.json()
                total_reward += step_data["reward"]["value"]
                obs = step_data["observation"]
                steps += 1
                if step_data["done"]:
                    break

            while not step_data["done"]:
                action = {"action_type": "hold"}
                res = client.post(f"/step?session_id={session_id}", json=action)
                step_data = res.json()
                total_reward += step_data["reward"]["value"]
                obs = step_data["observation"]
                steps += 1

        else:  # hard
            actions_init = [
                {"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.15},
                {"action_type": "add_drug", "drug": "insulin", "dose": 5.0},
            ]
            for action in actions_init:
                res = client.post(f"/step?session_id={session_id}", json=action)
                step_data = res.json()
                total_reward += step_data["reward"]["value"]
                obs = step_data["observation"]
                steps += 1
                if step_data["done"]:
                    break

            while not step_data["done"]:
                action = {"action_type": "hold"}
                res = client.post(f"/step?session_id={session_id}", json=action)
                step_data = res.json()
                total_reward += step_data["reward"]["value"]
                obs = step_data["observation"]
                steps += 1

        # Grade
        res = client.get(f"/grader?session_id={session_id}")
        grade = res.json()
        results[task_id] = {
            "score": grade["score"],
            "total_reward": round(total_reward, 4),
            "steps": steps,
            "done_reason": grade.get("done_reason"),
            "breakdown": grade["breakdown"],
        }
        print(f"Score: {grade['score']:.4f} | Reward: {total_reward:.4f} | Steps: {steps}")

    client.close()
    return results


def run_llm_baseline(server_url: str, model: str = "gpt-4o-mini") -> dict:
    """
    Run an LLM-powered baseline using the OpenAI API.

    The LLM receives the observation and decides actions.
    """
    if not HAS_OPENAI:
        print("OpenAI package not installed. Install with: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    http_client = httpx.Client(base_url=server_url, timeout=30.0)
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Running LLM baseline ({model}): {task_id}")
        print(f"{'='*50}")

        # Reset
        res = http_client.post("/reset", json={"task_id": task_id, "seed": 42})
        data = res.json()
        session_id = data["session_id"]
        obs = data["observation"]

        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Format observation for LLM
            obs_text = json.dumps(obs, indent=2)
            user_msg = f"Current observation (step {obs['current_step']}/{obs['max_steps']}):\n{obs_text}\n\nDecide your action:"

            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,  # Deterministic
                    max_tokens=200,
                )
                action_str = response.choices[0].message.content.strip()
                # Parse JSON from response
                if "```" in action_str:
                    action_str = action_str.split("```")[1]
                    if action_str.startswith("json"):
                        action_str = action_str[4:]
                action = json.loads(action_str)
            except Exception as e:
                print(f"  LLM error at step {steps}: {e}")
                action = {"action_type": "hold"}

            # Execute step
            res = http_client.post(
                f"/step?session_id={session_id}",
                json=action,
            )
            step_data = res.json()
            total_reward += step_data["reward"]["value"]
            obs = step_data["observation"]
            done = step_data["done"]
            steps += 1
            print(f"  Step {steps}: {action.get('action_type')} → R={step_data['reward']['value']:.3f}")

        # Grade
        res = http_client.get(f"/grader?session_id={session_id}")
        grade = res.json()
        results[task_id] = {
            "score": grade["score"],
            "total_reward": round(total_reward, 4),
            "steps": steps,
            "done_reason": grade.get("done_reason"),
            "breakdown": grade["breakdown"],
        }
        print(f"Score: {grade['score']:.4f} | Reward: {total_reward:.4f}")

    http_client.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="ICU Drug Titration Baseline")
    parser.add_argument("--server-url", default="http://localhost:7860", help="Server URL")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic",
                        help="Baseline mode: heuristic (no API key) or llm (requires OpenAI)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for LLM mode")
    args = parser.parse_args()

    print("=" * 60)
    print("ICU Drug Titration Environment — Baseline Evaluation")
    print("=" * 60)

    if args.mode == "llm":
        results = run_llm_baseline(args.server_url, args.model)
    else:
        results = run_heuristic_baseline(args.server_url)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for task, data in results.items():
        print(f"  {task:8s}: score={data['score']:.4f}  reward={data['total_reward']:.4f}  steps={data['steps']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
