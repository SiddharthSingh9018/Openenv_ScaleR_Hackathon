
from typing import List, Dict

def grade_task_1_static(episode_log: List[Dict]) -> float:
	if not episode_log:
		return 0.0
	collision = any(e["collision"] for e in episode_log)
	max_vehicle_x = max(e["vehicle_x"] for e in episode_log)
	vehicle_crossed = any(e["vehicle_x"] > 35.0 for e in episode_log)
	if collision:
		return 0.0
	if not vehicle_crossed:
		return min(0.4, max_vehicle_x / 35.0 * 0.4)
	base = 0.6
	# Speed bonus: average speed in crossing zone (x in [30,36])
	speeds = [e["speed"] for e in episode_log if 30.0 <= e["vehicle_x"] <= 36.0]
	avg_speed = sum(speeds)/len(speeds) if speeds else 0.0
	speed_bonus = min(0.2, avg_speed / 8.0 * 0.2)
	# Smoothness: jerk count (action changes accel→brake or brake→accel)
	actions = [e["action"] for e in episode_log]
	jerk_count = sum(
		(a1.startswith("SOFT_ACCEL") or a1.startswith("STRONG_ACCEL")) != (a2.startswith("SOFT_ACCEL") or a2.startswith("STRONG_ACCEL"))
		for a1, a2 in zip(actions, actions[1:])
	)
	smoothness_bonus = max(0.0, 0.2 - jerk_count * 0.04)
	return min(1.0, base + speed_bonus + smoothness_bonus)

def grade_task_2_stochastic(episode_log: List[Dict]) -> float:
	if not episode_log:
		return 0.0
	collision = any(e["collision"] for e in episode_log)
	max_x = max(e["vehicle_x"] for e in episode_log)
	vehicle_crossed = any(e["vehicle_x"] > 35.0 for e in episode_log)
	if collision:
		return 0.0
	if not vehicle_crossed:
		return min(0.3, max_x / 35.0 * 0.3)
	base = 0.5
	steps = len(episode_log)
	efficiency_bonus = max(0.0, 1 - steps/80) * 0.25
	# Belief-action correctness proxy
	correct = 0
	for e in episode_log:
		intent = e["intent"]
		action = e["action"]
		if intent == "CROSSING" and action in {"STRONG_BRAKE", "SOFT_BRAKE", "COAST"}:
			correct += 1
		elif intent == "RETREATING" and action in {"SOFT_ACCEL", "STRONG_ACCEL", "COAST"}:
			correct += 1
		elif intent == "HESITATING" and action in {"SOFT_BRAKE", "COAST", "SOFT_ACCEL"}:
			correct += 1
	belief_acc_bonus = (correct / steps) * 0.25 if steps else 0.0
	return min(1.0, base + efficiency_bonus + belief_acc_bonus)

def grade_task_3_adversarial(episode_log: List[Dict]) -> float:
	if not episode_log:
		return 0.0
	collision = any(e["collision"] for e in episode_log)
	vehicle_crossed = any(e["vehicle_x"] > 35.0 for e in episode_log)
	if collision:
		return 0.0
	if not vehicle_crossed:
		return 0.1
	base = 0.4
	steps = len(episode_log)
	efficiency = max(0.0, 1 - steps/80) * 0.2
	# Belief-action correctness proxy
	correct = 0
	for e in episode_log:
		intent = e["intent"]
		action = e["action"]
		if intent == "CROSSING" and action in {"STRONG_BRAKE", "SOFT_BRAKE", "COAST"}:
			correct += 1
		elif intent == "RETREATING" and action in {"SOFT_ACCEL", "STRONG_ACCEL", "COAST"}:
			correct += 1
		elif intent == "HESITATING" and action in {"SOFT_BRAKE", "COAST", "SOFT_ACCEL"}:
			correct += 1
	belief_acc = (correct / steps) * 0.2 if steps else 0.0
	# Oscillation: brake→accel→brake pattern
	actions = [e["action"] for e in episode_log]
	oscillation_count = 0
	for i in range(2, len(actions)):
		a0, a1, a2 = actions[i-2:i+1]
		if (a0.endswith("BRAKE") and a1.endswith("ACCEL") and a2.endswith("BRAKE")) or (a0.endswith("ACCEL") and a1.endswith("BRAKE") and a2.endswith("ACCEL")):
			oscillation_count += 1
	smoothness = max(0.0, 0.2 - oscillation_count * 0.025)
	return min(1.0, base + efficiency + belief_acc + smoothness)

GRADERS = {
	"task_1_static": grade_task_1_static,
	"task_2_stochastic": grade_task_2_stochastic,
	"task_3_adversarial": grade_task_3_adversarial,
}
