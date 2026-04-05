import random
from enum import Enum
from typing import Dict, List, Tuple

from pydantic import BaseModel

# Constants
ROAD_WIDTH = 6.0
CROSSING_Y = 30.0
DT = 0.5
MAX_STEPS = 80


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class Action(str, Enum):
    STRONG_BRAKE = "STRONG_BRAKE"
    SOFT_BRAKE = "SOFT_BRAKE"
    COAST = "COAST"
    SOFT_ACCEL = "SOFT_ACCEL"
    STRONG_ACCEL = "STRONG_ACCEL"


ACTION_TO_ACCEL = {
    Action.STRONG_BRAKE: -4.0,
    Action.SOFT_BRAKE: -2.0,
    Action.COAST: 0.0,
    Action.SOFT_ACCEL: 2.0,
    Action.STRONG_ACCEL: 4.0,
}


class PedIntent(str, Enum):
    CROSSING = "CROSSING"
    HESITATING = "HESITATING"
    RETREATING = "RETREATING"


class Observation(BaseModel):
    vehicle_x: float
    vehicle_speed: float
    dist_to_crossing: float
    time_to_crossing: float
    ped_x: float
    ped_y: float
    ped_vx: float
    belief_crossing: float
    belief_hesitating: float
    belief_retreating: float
    step: int
    collision: bool
    crossed: bool
    vehicle_crossed: bool


class ActionModel(BaseModel):
    action: Action


class Reward(BaseModel):
    total: float
    safety: float
    efficiency: float
    smoothness: float
    belief_accuracy: float


class PedestrianNegotiationEnv:
    TASKS = ["task_1_static", "task_2_stochastic", "task_3_adversarial"]

    def __init__(self, task: str, seed: int = 42):
        assert task in self.TASKS, f"Unknown task: {task}"
        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)
        self._episode_log: List[Dict] = []
        self.reset()

    def reset(self) -> Observation:
        self._rng.seed(self.seed)
        self.step_count = 0
        self.done = False
        self.vehicle_x = 0.0
        self.vehicle_speed = 5.0
        self.prev_accel = 0.0
        self.collision = False
        self.crossed = False
        self.vehicle_crossed = False
        self._episode_log = []
        self._prev_dist_to_crossing = CROSSING_Y
        self._idle_steps = 0
        self._ped_pause_steps = 0

        if self.task == "task_1_static":
            self.ped_x = -0.5
            self.ped_y = CROSSING_Y
            self.ped_vx = 0.0
            self.ped_intent = PedIntent.HESITATING
        elif self.task == "task_2_stochastic":
            self.ped_x = -0.5
            self.ped_y = CROSSING_Y
            self.ped_vx = 1.2
            self.ped_intent = PedIntent.CROSSING
        else:
            self.ped_x = -0.35
            self.ped_y = CROSSING_Y
            self.ped_vx = 1.0
            self.ped_intent = PedIntent.CROSSING

        self._belief = {
            PedIntent.CROSSING: 1 / 3,
            PedIntent.HESITATING: 1 / 3,
            PedIntent.RETREATING: 1 / 3,
        }
        return self._make_obs()

    def _noisy_value(self, value: float, sigma: float) -> float:
        return value + self._rng.gauss(0.0, sigma)

    def _make_obs(self) -> Observation:
        dist_to_crossing = max(0.0, CROSSING_Y - self.vehicle_x)
        time_to_crossing = dist_to_crossing / max(self.vehicle_speed, 0.1)
        return Observation(
            vehicle_x=self.vehicle_x,
            vehicle_speed=_clamp(self.vehicle_speed, 0.0, 15.0),
            dist_to_crossing=dist_to_crossing,
            time_to_crossing=time_to_crossing,
            ped_x=self._noisy_value(self.ped_x, 0.12),
            ped_y=self._noisy_value(self.ped_y, 0.08),
            ped_vx=self._noisy_value(self.ped_vx, 0.08),
            belief_crossing=self._belief[PedIntent.CROSSING],
            belief_hesitating=self._belief[PedIntent.HESITATING],
            belief_retreating=self._belief[PedIntent.RETREATING],
            step=self.step_count,
            collision=self.collision,
            crossed=self.crossed,
            vehicle_crossed=self.vehicle_crossed,
        )

    def _update_task_2_pedestrian(self) -> None:
        if self.ped_intent == PedIntent.CROSSING:
            self.ped_x += self.ped_vx * DT + self._rng.gauss(0.0, 0.04)
            if self.ped_x >= ROAD_WIDTH + 0.5:
                self.ped_x = ROAD_WIDTH + 0.5
                self.ped_intent = PedIntent.RETREATING
                self.ped_vx = -1.0
        elif self.ped_intent == PedIntent.RETREATING:
            self.ped_x += self.ped_vx * DT + self._rng.gauss(0.0, 0.04)
            if self.ped_x <= -0.5:
                self.ped_x = -0.5
                self.ped_intent = PedIntent.HESITATING
                self.ped_vx = 0.0
                self._ped_pause_steps = 0
        else:
            self._ped_pause_steps += 1
            if self._ped_pause_steps >= 2 and self._rng.random() < 0.18:
                self.ped_intent = PedIntent.CROSSING
                self.ped_vx = 1.1
                self._ped_pause_steps = 0

    def _update_task_3_pedestrian(self) -> None:
        dist_to_crossing = CROSSING_Y - self.vehicle_x
        vehicle_arrival = dist_to_crossing / max(self.vehicle_speed, 0.1)
        is_vehicle_pressing = dist_to_crossing < 18.0 and self.vehicle_speed > 2.0
        near_entry = self.ped_x < 1.8

        if self.ped_intent == PedIntent.CROSSING:
            self.ped_x += self.ped_vx * DT + self._rng.gauss(0.0, 0.06)
            if near_entry and is_vehicle_pressing and vehicle_arrival < 4.0 and self._rng.random() < 0.6:
                self.ped_intent = PedIntent.HESITATING
                self.ped_vx = 0.0
                self._ped_pause_steps = 0
            elif self.ped_x >= ROAD_WIDTH + 0.5:
                self.ped_x = ROAD_WIDTH + 0.5
                self.ped_intent = PedIntent.RETREATING
                self.ped_vx = -1.2
                self._ped_pause_steps = 0
        elif self.ped_intent == PedIntent.HESITATING:
            self._ped_pause_steps += 1
            if is_vehicle_pressing and self._rng.random() < 0.22:
                self.ped_intent = PedIntent.RETREATING
                self.ped_vx = -0.9
            elif vehicle_arrival > 4.5 or self.vehicle_speed < 1.0:
                resume_prob = 0.25 + min(0.35, 0.05 * self._ped_pause_steps)
                if self._rng.random() < resume_prob:
                    self.ped_intent = PedIntent.CROSSING
                    self.ped_vx = 1.2
                    self._ped_pause_steps = 0
            elif self._ped_pause_steps >= 3 and self._rng.random() < 0.12:
                self.ped_intent = PedIntent.CROSSING
                self.ped_vx = 1.0
                self._ped_pause_steps = 0
        else:
            self.ped_x += self.ped_vx * DT + self._rng.gauss(0.0, 0.03)
            if self.ped_x <= -0.5:
                self.ped_x = -0.5
                self.ped_intent = PedIntent.HESITATING
                self.ped_vx = 0.0
                self._ped_pause_steps = 0
            elif self.vehicle_speed < 0.8 and dist_to_crossing > 10.0 and self._rng.random() < 0.08:
                self.ped_intent = PedIntent.CROSSING
                self.ped_vx = 1.3
                self._ped_pause_steps = 0

    def _update_belief(self, prev_ped_x: float, prev_ped_vx: float) -> None:
        movement = self.ped_x - prev_ped_x
        accelerating_into_road = movement > 0.08 or self.ped_vx > 0.35
        retreating_motion = movement < -0.08 or self.ped_vx < -0.25
        stationary = abs(movement) < 0.05 and abs(self.ped_vx) < 0.12
        near_curb = self.ped_x < 0.6
        in_road = 0.0 <= self.ped_x <= ROAD_WIDTH
        direction_flip = prev_ped_vx > 0.15 and self.ped_vx < -0.15

        evidence = {
            PedIntent.CROSSING: 1.0,
            PedIntent.HESITATING: 1.0,
            PedIntent.RETREATING: 1.0,
        }

        if accelerating_into_road:
            evidence[PedIntent.CROSSING] += 1.8
        if in_road:
            evidence[PedIntent.CROSSING] += 0.8
        if stationary:
            evidence[PedIntent.HESITATING] += 1.6
        if near_curb:
            evidence[PedIntent.HESITATING] += 0.4
        if self._ped_pause_steps >= 2:
            evidence[PedIntent.HESITATING] += 0.3 + 0.1 * min(self._ped_pause_steps, 4)
        if retreating_motion:
            evidence[PedIntent.RETREATING] += 2.0
        if direction_flip:
            evidence[PedIntent.RETREATING] += 0.8
        if self.ped_x < -0.1:
            evidence[PedIntent.RETREATING] += 0.4

        if not accelerating_into_road:
            evidence[PedIntent.CROSSING] *= 0.75
        if not stationary:
            evidence[PedIntent.HESITATING] *= 0.8
        if not retreating_motion:
            evidence[PedIntent.RETREATING] *= 0.8

        for intent in self._belief:
            self._belief[intent] = max(1e-6, self._belief[intent] * evidence[intent])

        total = sum(self._belief.values())
        for intent in self._belief:
            self._belief[intent] /= total

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        accel = ACTION_TO_ACCEL[action]
        prev_ped_x = self.ped_x
        prev_ped_vx = self.ped_vx

        self.vehicle_speed = _clamp(self.vehicle_speed + accel * DT, 0.0, 15.0)
        self.vehicle_x += self.vehicle_speed * DT

        if self.task == "task_2_stochastic":
            self._update_task_2_pedestrian()
        elif self.task == "task_3_adversarial":
            self._update_task_3_pedestrian()

        self.ped_x = _clamp(self.ped_x, -0.5, ROAD_WIDTH + 0.5)
        ped_in_road = 0.0 <= self.ped_x <= ROAD_WIDTH
        dist_to_crossing = abs(self.vehicle_x - CROSSING_Y)

        self.collision = (
            ped_in_road
            and abs(self.ped_y - CROSSING_Y) < 1.0
            and abs(self.vehicle_x - CROSSING_Y) < 1.5
            and abs(self.ped_x - ROAD_WIDTH / 2) < 2.0
            and self.vehicle_speed > 0.5
        )
        self.crossed = self.ped_x >= ROAD_WIDTH + 0.5
        self.vehicle_crossed = self.vehicle_x >= CROSSING_Y + 5.0

        self._update_belief(prev_ped_x, prev_ped_vx)

        step_penalty = -0.02
        safety = -25.0 if self.collision else 0.0
        if ped_in_road and self.vehicle_x <= CROSSING_Y and self.vehicle_speed > 0.1:
            time_to_crossing = (CROSSING_Y - self.vehicle_x) / self.vehicle_speed
            if time_to_crossing < 1.2:
                safety -= 1.2
            elif time_to_crossing < 2.5:
                safety -= 0.45
            if self.ped_intent == PedIntent.CROSSING and self.vehicle_speed > 4.0:
                safety -= 0.2

        progress = self._prev_dist_to_crossing - dist_to_crossing
        progress_reward = max(-0.05, 0.05 * progress)
        self._prev_dist_to_crossing = dist_to_crossing

        if self.vehicle_speed < 0.2 and not ped_in_road:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        idle_penalty = -0.12 if self._idle_steps >= 3 else 0.0

        efficiency = 0.015 * self.vehicle_speed + step_penalty
        if self.vehicle_crossed and not self.collision:
            efficiency += 5.5

        switched_direction = (accel > 0 > self.prev_accel) or (accel < 0 < self.prev_accel)
        smoothness = -0.03 * abs(accel - self.prev_accel)
        if switched_direction:
            smoothness -= 0.12

        belief_accuracy = 0.1 * self._belief[self.ped_intent]
        total = safety + efficiency + smoothness + belief_accuracy + progress_reward + idle_penalty
        reward = Reward(
            total=float(total),
            safety=float(safety),
            efficiency=float(efficiency + progress_reward + idle_penalty),
            smoothness=float(smoothness),
            belief_accuracy=float(belief_accuracy),
        )
        self.prev_accel = accel

        self._episode_log.append(
            {
                "step": self.step_count,
                "action": action.value,
                "reward": float(total),
                "collision": self.collision,
                "vehicle_x": self.vehicle_x,
                "speed": self.vehicle_speed,
                "ped_x": self.ped_x,
                "intent": self.ped_intent.value,
            }
        )

        self.step_count += 1
        if self.collision or self.vehicle_crossed or self.step_count >= MAX_STEPS:
            self.done = True

        return self._make_obs(), reward, self.done, {}

    def state(self) -> dict:
        dist_to_crossing = max(0.0, CROSSING_Y - self.vehicle_x)
        time_to_crossing = dist_to_crossing / max(self.vehicle_speed, 0.1)
        return {
            "vehicle_x": self.vehicle_x,
            "vehicle_speed": self.vehicle_speed,
            "dist_to_crossing": dist_to_crossing,
            "time_to_crossing": time_to_crossing,
            "ped_x": self.ped_x,
            "ped_y": self.ped_y,
            "ped_vx": self.ped_vx,
            "step": self.step_count,
            "collision": self.collision,
            "crossed": self.crossed,
            "vehicle_crossed": self.vehicle_crossed,
            "true_intent": self.ped_intent.value,
            "belief": {intent.value: prob for intent, prob in self._belief.items()},
            "_episode_log": self._episode_log,
        }

    @property
    def episode_log(self):
        return self._episode_log
