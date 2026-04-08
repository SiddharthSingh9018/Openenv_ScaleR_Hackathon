import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT / "pedestrian-negotiation-env"
APP_DIR = PROJECT_ROOT / "app"
BASELINE_DIR = PROJECT_ROOT / "baseline"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from run_baseline import VALID_ACTIONS, select_rule_action  # noqa: E402

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["API_KEY"]
BENCHMARK = os.environ.get("BENCHMARK_NAME", "pedestrian-negotiation")
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "80"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.1"))
RESULTS_PATH = REPO_ROOT / "baseline_results.json"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def check_server() -> bool:
    try:
        response = httpx.get(f"{SERVER_URL}/", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def start_local_server() -> Optional[subprocess.Popen]:
    if check_server():
        return None
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
        ],
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


def warmup_proxy(client: OpenAI) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply with exactly one token: COAST"},
            {"role": "user", "content": "Return one valid action."},
        ],
        temperature=0,
        max_tokens=4,
    )


def llm_action(client: OpenAI, task_id: str, obs: Dict, step: int) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You control an autonomous vehicle at a crosswalk. "
                    "Infer whether the pedestrian is crossing, hesitating, or retreating. "
                    "Avoid collision, cross efficiently, and avoid oscillatory control. "
                    "Reply with exactly one action from: STRONG_BRAKE, SOFT_BRAKE, COAST, SOFT_ACCEL, STRONG_ACCEL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"task={task_id}, step={step}, vehicle_x={obs['vehicle_x']:.2f}, "
                    f"vehicle_speed={obs['vehicle_speed']:.2f}, "
                    f"dist_to_crossing={obs.get('dist_to_crossing', max(0.0, 30.0 - obs['vehicle_x'])):.2f}, "
                    f"time_to_crossing={obs.get('time_to_crossing', max(0.0, 30.0 - obs['vehicle_x']) / max(obs['vehicle_speed'], 0.1)):.2f}, "
                    f"ped_x={obs['ped_x']:.2f}, ped_vx={obs['ped_vx']:.2f}, "
                    f"belief_crossing={obs['belief_crossing']:.2f}, "
                    f"belief_hesitating={obs['belief_hesitating']:.2f}, "
                    f"belief_retreating={obs['belief_retreating']:.2f}"
                ),
            },
        ],
        temperature=0,
        max_tokens=8,
    )
    content = (response.choices[0].message.content or "").strip()
    action = content.split()[0] if content else "COAST"
    return action if action in VALID_ACTIONS else "COAST"


def run_episode(task: Dict, client: Optional[OpenAI]) -> Dict:
    task_id = task["id"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset = httpx.post(
            f"{SERVER_URL}/reset",
            json={"task": task_id, "seed": DEFAULT_SEED},
            timeout=15.0,
        )
        reset.raise_for_status()
        obs = reset.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            action = select_rule_action(task_id, obs)
            if client is not None:
                try:
                    action = llm_action(client, task_id, obs, step)
                    last_error = None
                except Exception as exc:
                    action = select_rule_action(task_id, obs)
                    last_error = str(exc)

            step_response = httpx.post(
                f"{SERVER_URL}/step",
                json={"task": task_id, "action": action},
                timeout=15.0,
            )
            step_response.raise_for_status()
            data = step_response.json()
            obs = data["observation"]
            done = data["done"]
            reward = float(data["reward"]["total"])

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=last_error)

            if done:
                break

        grade = httpx.post(f"{SERVER_URL}/grader", json={"task": task_id}, timeout=15.0)
        grade.raise_for_status()
        grade_json = grade.json()
        score = float(grade_json["score"])
        success = score >= SUCCESS_SCORE_THRESHOLD
        return {
            "score": score,
            "steps": steps_taken,
            "collision": bool(grade_json["details"]["collision"]),
        }
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> int:
    server_process = start_local_server()
    if not check_server():
        return 1

    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_BASE_URL"] = API_BASE_URL
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    warmup_proxy(client)

    try:
        tasks = get_tasks()
        results = {}
        for task in tasks:
            results[task["id"]] = run_episode(task, client)

        with RESULTS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        return 0
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except Exception:
                server_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
