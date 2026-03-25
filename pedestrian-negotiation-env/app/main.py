
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import sys
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

@app.get("/")
def root():
	return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest):
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
		while not done and steps < 80:
			# Rule-based agent
			ped_in_road = 0.0 <= obs.ped_x <= 6.0
			dist = abs(obs.vehicle_x - 30.0)
			if ped_in_road and dist < 10.0:
				action = Action.STRONG_BRAKE
			elif ped_in_road and dist < 20.0:
				action = Action.SOFT_BRAKE
			elif obs.ped_x >= 6.0 or obs.belief_retreating > 0.6:
				action = Action.SOFT_ACCEL
			else:
				action = Action.COAST
			obs, reward, done, _ = env.step(action)
			steps += 1
		fn = GRADERS[t.id]
		score = fn(env.episode_log)
		results[t.id] = {"score": score, "steps": steps, "collision": any(e["collision"] for e in env.episode_log)}
	return results
