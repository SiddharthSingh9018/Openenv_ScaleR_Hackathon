
import os
import sys
import time
import json
import httpx

SERVER = os.environ.get("SERVER_URL", "http://localhost:7860")
TASKS = [
	{"id": "task_1_static", "desc": "Static pedestrian"},
	{"id": "task_2_stochastic", "desc": "Stochastic crossing"},
	{"id": "task_3_adversarial", "desc": "Adversarial"},
]

def check_server():
	try:
		r = httpx.get(f"{SERVER}/", timeout=3)
		return r.status_code == 200
	except Exception:
		return False

def get_tasks():
	r = httpx.get(f"{SERVER}/tasks")
	return r.json()

def run_rule_based(task):
	r = httpx.post(f"{SERVER}/reset", json={"task": task, "seed": 42})
	obs = r.json()
	steps = 0
	done = False
	while not done and steps < 80:
		ped_in_road = 0.0 <= obs["ped_x"] <= 6.0
		dist = abs(obs["vehicle_x"] - 30.0)
		if ped_in_road and dist < 10.0:
			action = "STRONG_BRAKE"
		elif ped_in_road and dist < 20.0:
			action = "SOFT_BRAKE"
		elif obs["ped_x"] >= 6.0 or obs["belief_retreating"] > 0.6:
			action = "SOFT_ACCEL"
		else:
			action = "COAST"
		r = httpx.post(f"{SERVER}/step", json={"task": task, "action": action})
		data = r.json()
		obs = data["observation"]
		done = data["done"]
		steps += 1
		if done:
			break
	r = httpx.post(f"{SERVER}/grader", json={"task": task})
	score = r.json()["score"]
	details = r.json()["details"]
	return {"score": score, "steps": steps, "collision": details["collision"]}

def run_llm_agent(task):
	import openai
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		return None
	openai.api_key = api_key
	r = httpx.post(f"{SERVER}/reset", json={"task": task, "seed": 42})
	obs = r.json()
	steps = 0
	done = False
	messages = [
		{"role": "system", "content": "You control an autonomous vehicle. A pedestrian may be crossing ahead.\nAvoid collision. Cross efficiently. Maintain smooth control.\nActions: STRONG_BRAKE, SOFT_BRAKE, COAST, SOFT_ACCEL, STRONG_ACCEL\nReply with ONLY the action name."}
	]
	while not done and steps < 80:
		user_msg = f"vehicle_x: {obs['vehicle_x']:.2f}, vehicle_speed: {obs['vehicle_speed']:.2f}, ped_x: {obs['ped_x']:.2f}, dist_to_crossing: {abs(obs['vehicle_x']-30.0):.2f}, belief_crossing: {obs['belief_crossing']:.2f}, belief_hesitating: {obs['belief_hesitating']:.2f}, belief_retreating: {obs['belief_retreating']:.2f}, step: {obs['step']}"
		messages.append({"role": "user", "content": user_msg})
		resp = openai.chat.completions.create(
			model="gpt-4o-mini",
			messages=messages,
			temperature=0,
			max_tokens=8,
		)
		action = resp.choices[0].message.content.strip().split()[0]
		if action not in {"STRONG_BRAKE", "SOFT_BRAKE", "COAST", "SOFT_ACCEL", "STRONG_ACCEL"}:
			action = "COAST"
		r = httpx.post(f"{SERVER}/step", json={"task": task, "action": action})
		data = r.json()
		obs = data["observation"]
		done = data["done"]
		steps += 1
		if done:
			break
	r = httpx.post(f"{SERVER}/grader", json={"task": task})
	score = r.json()["score"]
	details = r.json()["details"]
	return {"score": score, "steps": steps, "collision": details["collision"]}

def main():
	print(f"Checking server at {SERVER} ...")
	if not check_server():
		print("Server not running. Start FastAPI server first.")
		sys.exit(1)
	print("Server is up!")
	print("Task list:")
	tasks = get_tasks()
	for t in tasks:
		print(f"- {t['id']}: {t['description']}")
	print("\nRunning rule-based agent...")
	results = {}
	for t in TASKS:
		res = run_rule_based(t["id"])
		print(f"{t['id']}: score={res['score']:.3f}, steps={res['steps']}, collision={res['collision']}")
		results[t["id"]] = res
	if os.environ.get("OPENAI_API_KEY"):
		print("\nRunning LLM agent (gpt-4o-mini)...")
		for t in TASKS:
			res = run_llm_agent(t["id"])
			if res:
				print(f"LLM {t['id']}: score={res['score']:.3f}, steps={res['steps']}, collision={res['collision']}")
				results[f"llm_{t['id']}"] = res
	print("\nSummary:")
	for k, v in results.items():
		print(f"{k}: score={v['score']:.3f}, steps={v['steps']}, collision={v['collision']}")
	with open("baseline_results.json", "w") as f:
		json.dump(results, f, indent=2)
	print("Results saved to baseline_results.json")

if __name__ == "__main__":
	main()
