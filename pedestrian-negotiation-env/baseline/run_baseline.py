import json
import os
import sys

import httpx

SERVER = os.environ.get("SERVER_URL", "http://localhost:7860")
TASKS = [
    {"id": "task_1_static", "desc": "Static pedestrian"},
    {"id": "task_2_stochastic", "desc": "Stochastic crossing"},
    {"id": "task_3_adversarial", "desc": "Adversarial"},
]
VALID_ACTIONS = {
    "STRONG_BRAKE",
    "SOFT_BRAKE",
    "COAST",
    "SOFT_ACCEL",
    "STRONG_ACCEL",
}


def check_server():
    try:
        response = httpx.get(f"{SERVER}/", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_tasks():
    response = httpx.get(f"{SERVER}/tasks")
    return response.json()


def select_rule_action(task, obs):
    dist_to_crossing = max(0.0, 30.0 - obs["vehicle_x"])
    vehicle_speed = max(obs["vehicle_speed"], 0.1)
    time_to_crossing = dist_to_crossing / vehicle_speed
    ped_x = obs["ped_x"]
    ped_vx = obs["ped_vx"]
    ped_in_road = -0.1 <= ped_x <= 6.1
    ped_advancing = ped_vx > 0.2 or obs["belief_crossing"] > 0.52
    ped_retreating = ped_vx < -0.2 or obs["belief_retreating"] > 0.58 or ped_x >= 6.0
    cautious_bias = 0.35 if task == "task_3_adversarial" else 0.0

    if ped_in_road and time_to_crossing < 1.4 + cautious_bias:
        return "STRONG_BRAKE"
    if ped_advancing and ped_x > -0.15 and time_to_crossing < 2.5 + cautious_bias:
        return "SOFT_BRAKE"
    if ped_advancing and ped_x > 1.2 and time_to_crossing < 3.0 + cautious_bias:
        return "SOFT_BRAKE"
    if ped_retreating and dist_to_crossing > 4.0:
        return "STRONG_ACCEL" if task != "task_3_adversarial" else "SOFT_ACCEL"
    if task == "task_1_static" and ped_x < 0.0 and obs["belief_hesitating"] > 0.55:
        return "STRONG_ACCEL"
    if obs["belief_hesitating"] > 0.55 and dist_to_crossing > 10.0:
        return "SOFT_ACCEL"
    if dist_to_crossing < 8.0 and not ped_in_road:
        return "STRONG_ACCEL"
    return "COAST"


def run_rule_based(task):
    response = httpx.post(f"{SERVER}/reset", json={"task": task, "seed": 42})
    obs = response.json()
    steps = 0
    done = False
    while not done and steps < 80:
        action = select_rule_action(task, obs)
        response = httpx.post(f"{SERVER}/step", json={"task": task, "action": action})
        data = response.json()
        obs = data["observation"]
        done = data["done"]
        steps += 1
    response = httpx.post(f"{SERVER}/grader", json={"task": task})
    result = response.json()
    return {"score": result["score"], "steps": steps, "collision": result["details"]["collision"]}


def run_llm_agent(task):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    response = httpx.post(f"{SERVER}/reset", json={"task": task, "seed": 42})
    obs = response.json()
    steps = 0
    done = False
    messages = [
        {
            "role": "system",
            "content": (
                "You control an autonomous vehicle at a crosswalk. "
                "Infer whether the pedestrian is crossing, hesitating, or retreating. "
                "Avoid collision, cross efficiently, and avoid oscillatory control. "
                "Actions: STRONG_BRAKE, SOFT_BRAKE, COAST, SOFT_ACCEL, STRONG_ACCEL. "
                "Reply with only the action name."
            ),
        }
    ]
    while not done and steps < 80:
        user_msg = (
            f"vehicle_x={obs['vehicle_x']:.2f}, vehicle_speed={obs['vehicle_speed']:.2f}, "
            f"ped_x={obs['ped_x']:.2f}, ped_vx={obs['ped_vx']:.2f}, "
            f"dist_to_crossing={max(0.0, 30.0 - obs['vehicle_x']):.2f}, "
            f"belief_crossing={obs['belief_crossing']:.2f}, "
            f"belief_hesitating={obs['belief_hesitating']:.2f}, "
            f"belief_retreating={obs['belief_retreating']:.2f}, step={obs['step']}"
        )
        messages.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=8,
        )
        action = resp.choices[0].message.content.strip().split()[0]
        if action not in VALID_ACTIONS:
            action = "COAST"
        response = httpx.post(f"{SERVER}/step", json={"task": task, "action": action})
        data = response.json()
        obs = data["observation"]
        done = data["done"]
        steps += 1
    response = httpx.post(f"{SERVER}/grader", json={"task": task})
    result = response.json()
    return {"score": result["score"], "steps": steps, "collision": result["details"]["collision"]}


def main():
    print(f"Checking server at {SERVER} ...")
    if not check_server():
        print("Server not running. Start FastAPI server first.")
        sys.exit(1)
    print("Server is up!")
    print("Task list:")
    for task in get_tasks():
        print(f"- {task['id']}: {task['description']}")

    print("\nRunning rule-based agent...")
    results = {}
    for task in TASKS:
        res = run_rule_based(task["id"])
        print(f"{task['id']}: score={res['score']:.3f}, steps={res['steps']}, collision={res['collision']}")
        results[task["id"]] = res

    if os.environ.get("OPENAI_API_KEY"):
        print("\nRunning LLM agent (gpt-4o-mini)...")
        for task in TASKS:
            res = run_llm_agent(task["id"])
            if res:
                print(
                    f"LLM {task['id']}: score={res['score']:.3f}, "
                    f"steps={res['steps']}, collision={res['collision']}"
                )
                results[f"llm_{task['id']}"] = res

    print("\nSummary:")
    for key, value in results.items():
        print(f"{key}: score={value['score']:.3f}, steps={value['steps']}, collision={value['collision']}")

    with open("baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
