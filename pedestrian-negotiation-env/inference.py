import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
BASELINE_DIR = PROJECT_ROOT / "baseline"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from run_baseline import VALID_ACTIONS, select_rule_action  # noqa: E402

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "80"))
RESULTS_PATH = Path.cwd() / "baseline_results.json"


def emit(tag: str, payload: Dict) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)


def check_server() -> bool:
    try:
        response = httpx.get(f"{SERVER_URL}/", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def start_local_server() -> Optional[subprocess.Popen]:
    if check_server():
        return None
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(APP_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if check_server():
            return process
        time.sleep(1)
    process.terminate()
    return None


def get_tasks() -> List[Dict]:
    response = httpx.get(f"{SERVER_URL}/tasks", timeout=10.0)
    response.raise_for_status()
    return response.json()


def llm_action(client: OpenAI, task_id: str, obs: Dict, step: int) -> str:
    system = (
        "You control an autonomous vehicle at a crosswalk. "
        "Infer whether the pedestrian is crossing, hesitating, or retreating. "
        "Avoid collision, cross efficiently, and avoid oscillatory control. "
        "Reply with only one action from: STRONG_BRAKE, SOFT_BRAKE, COAST, SOFT_ACCEL, STRONG_ACCEL."
    )
    user = (
        f"task={task_id}, step={step}, vehicle_x={obs['vehicle_x']:.2f}, "
        f"vehicle_speed={obs['vehicle_speed']:.2f}, dist_to_crossing={obs.get('dist_to_crossing', max(0.0, 30.0 - obs['vehicle_x'])):.2f}, "
        f"time_to_crossing={obs.get('time_to_crossing', max(0.0, 30.0 - obs['vehicle_x']) / max(obs['vehicle_speed'], 0.1)):.2f}, "
        f"ped_x={obs['ped_x']:.2f}, ped_vx={obs['ped_vx']:.2f}, "
        f"belief_crossing={obs['belief_crossing']:.2f}, belief_hesitating={obs['belief_hesitating']:.2f}, "
        f"belief_retreating={obs['belief_retreating']:.2f}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=8,
    )
    action = response.choices[0].message.content.strip().split()[0]
    return action if action in VALID_ACTIONS else "COAST"


def run_episode(task: Dict, client: Optional[OpenAI]) -> Dict:
    task_id = task["id"]
    reset = httpx.post(f"{SERVER_URL}/reset", json={"task": task_id, "seed": DEFAULT_SEED}, timeout=15.0)
    reset.raise_for_status()
    obs = reset.json()
    emit(
        "STEP",
        {
            "task": task_id,
            "step": 0,
            "event": "reset",
            "vehicle_x": round(obs["vehicle_x"], 4),
            "vehicle_speed": round(obs["vehicle_speed"], 4),
            "ped_x": round(obs["ped_x"], 4),
            "ped_vx": round(obs["ped_vx"], 4),
        },
    )

    done = False
    steps = 0
    llm_failures = 0
    while not done and steps < MAX_STEPS:
        action_source = "rule"
        action = select_rule_action(task_id, obs)
        if client is not None:
            try:
                action = llm_action(client, task_id, obs, steps)
                action_source = "llm"
            except Exception:
                llm_failures += 1
                action = select_rule_action(task_id, obs)
                action_source = "fallback_rule"

        step_response = httpx.post(
            f"{SERVER_URL}/step",
            json={"task": task_id, "action": action},
            timeout=15.0,
        )
        step_response.raise_for_status()
        data = step_response.json()
        obs = data["observation"]
        done = data["done"]
        steps += 1

        emit(
            "STEP",
            {
                "task": task_id,
                "step": steps,
                "event": "action",
                "action": action,
                "source": action_source,
                "vehicle_x": round(obs["vehicle_x"], 4),
                "vehicle_speed": round(obs["vehicle_speed"], 4),
                "ped_x": round(obs["ped_x"], 4),
                "ped_vx": round(obs["ped_vx"], 4),
                "done": done,
            },
        )

    grade = httpx.post(f"{SERVER_URL}/grader", json={"task": task_id}, timeout=15.0)
    grade.raise_for_status()
    result = grade.json()
    return {
        "score": result["score"],
        "steps": steps,
        "collision": result["details"]["collision"],
        "llm_failures": llm_failures,
    }


def main() -> int:
    emit(
        "START",
        {
            "server_url": SERVER_URL,
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "seed": DEFAULT_SEED,
        },
    )

    server_process = start_local_server()
    if not check_server():
        emit("END", {"status": "error", "reason": "server_unreachable"})
        return 1

    client = None
    if API_BASE_URL and MODEL_NAME and HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        tasks = get_tasks()
        results = {}
        for task in tasks:
            results[task["id"]] = run_episode(task, client)

        with RESULTS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        emit("END", {"status": "success", "results": results})
        return 0
    except Exception as exc:
        emit("END", {"status": "error", "reason": str(exc)})
        return 1
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except Exception:
                server_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
