
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import sys
import uvicorn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from environment import PedestrianNegotiationEnv, Action, ActionModel, Observation, Reward
from graders import GRADERS

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

_envs: Dict[str, PedestrianNegotiationEnv] = {}

class ResetRequest(BaseModel):
	task: str
	seed: int = 42

class StepRequest(BaseModel):
	task: str
	action: Action

class GraderRequest(BaseModel):
	task: str

class TaskInfo(BaseModel):
	id: str
	description: str
	difficulty: str
	max_steps: int

TASKS = [
	TaskInfo(id="task_1_static", description="Static pedestrian at road edge. Easy.", difficulty="easy", max_steps=80),
	TaskInfo(id="task_2_stochastic", description="Stochastic crossing pedestrian. Medium.", difficulty="medium", max_steps=80),
	TaskInfo(id="task_3_adversarial", description="Adversarial hesitating pedestrian. Hard.", difficulty="hard", max_steps=80),
]


def _select_baseline_action(task: str, obs: Observation) -> Action:
	dist_to_crossing = max(0.0, 30.0 - obs.vehicle_x)
	vehicle_speed = max(obs.vehicle_speed, 0.1)
	time_to_crossing = dist_to_crossing / vehicle_speed
	ped_in_road = -0.1 <= obs.ped_x <= 6.1
	ped_advancing = obs.ped_vx > 0.2 or obs.belief_crossing > 0.52
	ped_retreating = obs.ped_vx < -0.2 or obs.belief_retreating > 0.58 or obs.ped_x >= 6.0
	cautious_bias = 0.35 if task == "task_3_adversarial" else 0.0

	if ped_in_road and time_to_crossing < 1.4 + cautious_bias:
		return Action.STRONG_BRAKE
	if ped_advancing and obs.ped_x > -0.15 and time_to_crossing < 2.5 + cautious_bias:
		return Action.SOFT_BRAKE
	if ped_advancing and obs.ped_x > 1.2 and time_to_crossing < 3.0 + cautious_bias:
		return Action.SOFT_BRAKE
	if ped_retreating and dist_to_crossing > 4.0:
		return Action.SOFT_ACCEL if task == "task_3_adversarial" else Action.STRONG_ACCEL
	if task == "task_1_static" and obs.ped_x < 0.0 and obs.belief_hesitating > 0.55:
		return Action.STRONG_ACCEL
	if obs.belief_hesitating > 0.55 and dist_to_crossing > 10.0:
		return Action.SOFT_ACCEL
	if dist_to_crossing < 8.0 and not ped_in_road:
		return Action.STRONG_ACCEL
	return Action.COAST

@app.get("/")
def root():
	return {"status": "ok"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
	if req is None:
		req = ResetRequest(task=TASKS[0].id, seed=42)
	env = PedestrianNegotiationEnv(req.task, req.seed)
	_envs[req.task] = env
	obs = env.reset()
	return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
	env = _envs.get(req.task)
	if not env:
		raise HTTPException(400, "Call /reset first for this task.")
	try:
		obs, reward, done, info = env.step(req.action)
	except RuntimeError as e:
		raise HTTPException(400, str(e))
	return {
		"observation": obs.model_dump(),
		"reward": reward.model_dump(),
		"done": done,
		"info": info,
	}

@app.get("/state")
def state(task: str):
	env = _envs.get(task)
	if not env:
		raise HTTPException(400, "Call /reset first for this task.")
	return env.state()

@app.get("/tasks")
def get_tasks():
	return [t.model_dump() for t in TASKS]

@app.post("/grader")
def grader(req: GraderRequest):
	env = _envs.get(req.task)
	if not env:
		raise HTTPException(400, "Call /reset and run a full episode first.")
	log = env.episode_log
	if not log:
		raise HTTPException(400, "No episode log found.")
	fn = GRADERS.get(req.task)
	if not fn:
		raise HTTPException(400, "No grader for this task.")
	score = fn(log)
	return {"task": req.task, "score": score, "details": {"steps": len(log), "collision": any(e["collision"] for e in log)}}

@app.get("/baseline")
def baseline():
	results = {}
	for t in TASKS:
		env = PedestrianNegotiationEnv(t.id, 42)
		obs = env.reset()
		steps = 0
		done = False
		while not done and steps < t.max_steps:
			action = _select_baseline_action(t.id, obs)
			obs, _, done, _ = env.step(action)
			steps += 1
		fn = GRADERS[t.id]
		score = fn(env.episode_log)
		results[t.id] = {"score": score, "steps": steps, "collision": any(e["collision"] for e in env.episode_log)}
	return results


def main():
	uvicorn.run("main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
