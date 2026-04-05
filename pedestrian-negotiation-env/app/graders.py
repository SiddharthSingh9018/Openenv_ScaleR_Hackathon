from typing import Dict, List


def _crossed_successfully(episode_log: List[Dict], threshold: float = 35.0) -> bool:
    return any(event["vehicle_x"] >= threshold for event in episode_log)


def _action_alignment(intent: str, action: str) -> float:
    if intent == "CROSSING":
        if action.endswith("BRAKE"):
            return 1.0
        if action == "COAST":
            return 0.75
        return 0.0
    if intent == "RETREATING":
        if action.endswith("ACCEL"):
            return 1.0
        if action == "COAST":
            return 0.7
        return 0.0
    if intent == "HESITATING":
        if action in {"SOFT_BRAKE", "COAST", "SOFT_ACCEL"}:
            return 1.0
        return 0.35
    return 0.0


def _oscillation_count(actions: List[str]) -> int:
    oscillations = 0
    for i in range(2, len(actions)):
        a0, a1, a2 = actions[i - 2 : i + 1]
        if (a0.endswith("BRAKE") and a1.endswith("ACCEL") and a2.endswith("BRAKE")) or (
            a0.endswith("ACCEL") and a1.endswith("BRAKE") and a2.endswith("ACCEL")
        ):
            oscillations += 1
    return oscillations


def _hard_brake_count(actions: List[str]) -> int:
    return sum(action == "STRONG_BRAKE" for action in actions)


def grade_task_1_static(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    if any(event["collision"] for event in episode_log):
        return 0.0

    max_vehicle_x = max(event["vehicle_x"] for event in episode_log)
    if not _crossed_successfully(episode_log, threshold=34.0):
        return min(0.4, max_vehicle_x / 35.0 * 0.4)

    base = 0.6
    speeds = [event["speed"] for event in episode_log if 30.0 <= event["vehicle_x"] <= 36.0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    speed_bonus = min(0.2, avg_speed / 8.0 * 0.2)

    actions = [event["action"] for event in episode_log]
    jerk_count = sum(
        (a1.endswith("ACCEL")) != (a2.endswith("ACCEL")) for a1, a2 in zip(actions, actions[1:])
    )
    smoothness_bonus = max(0.0, 0.2 - jerk_count * 0.04)
    return min(1.0, base + speed_bonus + smoothness_bonus)


def grade_task_2_stochastic(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    if any(event["collision"] for event in episode_log):
        return 0.0

    max_x = max(event["vehicle_x"] for event in episode_log)
    if not _crossed_successfully(episode_log):
        return min(0.3, max_x / 35.0 * 0.3)

    steps = len(episode_log)
    base = 0.48
    efficiency_bonus = max(0.0, 1 - steps / 80) * 0.22
    alignment_bonus = sum(_action_alignment(event["intent"], event["action"]) for event in episode_log) / steps
    smooth_actions = [event["action"] for event in episode_log]
    oscillation_penalty = min(0.08, _oscillation_count(smooth_actions) * 0.02)
    return min(1.0, base + alignment_bonus * 0.3 + efficiency_bonus - oscillation_penalty)


def grade_task_3_adversarial(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    if any(event["collision"] for event in episode_log):
        return 0.0

    max_x = max(event["vehicle_x"] for event in episode_log)
    if not _crossed_successfully(episode_log):
        return min(0.25, max_x / 35.0 * 0.25)

    steps = len(episode_log)
    actions = [event["action"] for event in episode_log]
    oscillations = _oscillation_count(actions)
    hard_brakes = _hard_brake_count(actions)
    hesitation_events = sum(event["intent"] == "HESITATING" for event in episode_log)
    retreat_events = sum(event["intent"] == "RETREATING" for event in episode_log)

    base = 0.35
    efficiency_bonus = max(0.0, 1 - steps / 60) * 0.18
    alignment_bonus = sum(_action_alignment(event["intent"], event["action"]) for event in episode_log) / steps
    adversarial_coverage = min(1.0, (hesitation_events + retreat_events) / max(1, steps * 0.35))
    oscillation_penalty = min(0.2, oscillations * 0.04)
    overreaction_penalty = min(0.12, max(0, hard_brakes - 2) * 0.02)

    score = (
        base
        + alignment_bonus * 0.22
        + efficiency_bonus
        + adversarial_coverage * 0.15
        - oscillation_penalty
        - overreaction_penalty
    )
    return max(0.0, min(1.0, score))


GRADERS = {
    "task_1_static": grade_task_1_static,
    "task_2_stochastic": grade_task_2_stochastic,
    "task_3_adversarial": grade_task_3_adversarial,
}
