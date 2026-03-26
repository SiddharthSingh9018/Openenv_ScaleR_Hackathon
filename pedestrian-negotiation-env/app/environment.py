
import math
import random
from enum import Enum
from typing import Optional, Tuple, List, Dict
from pydantic import BaseModel, Field

# Constants
ROAD_WIDTH  = 6.0    # metres
CROSSING_Y  = 30.0   # crosswalk at 30m ahead
DT          = 0.5    # timestep seconds
MAX_STEPS   = 80

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

def _gauss(mu, sigma):
	return random.gauss(mu, sigma)

class PedestrianNegotiationEnv:
	TASKS = ["task_1_static", "task_2_stochastic", "task_3_adversarial"]

	def __init__(self, task: str, seed: int = 42):
		assert task in self.TASKS, f"Unknown task: {task}"
		self.task = task
		self.seed = seed
		self._rng = random.Random(seed)
		self._episode_log: List[Dict] = []
		self._vehicle_speed = 5.0  # BUG 2: set starting speed to 5.0
		self.reset()

	def reset(self) -> Observation:
		self._rng.seed(self.seed)
		self.step_count = 0
		self.done = False
		self.vehicle_x = 0.0
		self.vehicle_speed = 5.0  # BUG 2: set starting speed to 5.0
		self.prev_accel = 0.0
		self.collision = False
		self.crossed = False
		self.vehicle_crossed = False
		self._episode_log = []
		# Pedestrian state
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
		elif self.task == "task_3_adversarial":
			self.ped_x = -0.5
			self.ped_y = CROSSING_Y
			self.ped_vx = 1.2
			self.ped_intent = PedIntent.CROSSING
		self._belief = {
			PedIntent.CROSSING: 1/3,
			PedIntent.HESITATING: 1/3,
			PedIntent.RETREATING: 1/3,
		}
		return self._make_obs()

	def _make_obs(self) -> Observation:
		# Add noise to pedestrian obs
		ped_x_obs = self.ped_x + _gauss(0, 0.15)
		ped_y_obs = self.ped_y + _gauss(0, 0.10)
		ped_vx_obs = self.ped_vx + _gauss(0, 0.10)
		return Observation(
			vehicle_x=self.vehicle_x,
			vehicle_speed=max(0.0, min(self.vehicle_speed, 15.0)),
			ped_x=ped_x_obs,
			ped_y=ped_y_obs,
			ped_vx=ped_vx_obs,
			belief_crossing=self._belief[PedIntent.CROSSING],
			belief_hesitating=self._belief[PedIntent.HESITATING],
			belief_retreating=self._belief[PedIntent.RETREATING],
			step=self.step_count,
			collision=self.collision,
			crossed=self.crossed,
			vehicle_crossed=self.vehicle_crossed,
		)

	def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
		if self.done:
			raise RuntimeError("Episode is done. Call reset().")
		accel = ACTION_TO_ACCEL[action]
		# Vehicle dynamics
		self.vehicle_speed = max(0.0, min(self.vehicle_speed + accel * DT, 15.0))
		self.vehicle_x += self.vehicle_speed * DT
		# Pedestrian dynamics
		prev_ped_x = self.ped_x
		prev_intent = self.ped_intent
		if self.task == "task_1_static":
			pass  # Pedestrian does not move
		elif self.task == "task_2_stochastic":
			if self.ped_intent == PedIntent.CROSSING:
				self.ped_x += self.ped_vx * DT + _gauss(0, 0.05)
				if self.ped_x >= ROAD_WIDTH + 0.5:
					self.ped_intent = PedIntent.RETREATING
					self.ped_vx = -1.2
			elif self.ped_intent == PedIntent.RETREATING:
				self.ped_x += self.ped_vx * DT + _gauss(0, 0.05)
				if self.ped_x <= -0.5:
					self.ped_intent = PedIntent.HESITATING
					self.ped_vx = 0.0
		elif self.task == "task_3_adversarial":
			# Adversarial intent switching
			dist = abs(self.vehicle_x - CROSSING_Y)
			if self.ped_intent == PedIntent.CROSSING:
				if dist < 12.0 and self.vehicle_speed > 3.0:
					if self._rng.random() < 0.6:
						self.ped_intent = PedIntent.HESITATING
						self.ped_vx = 0.0
				else:
					self.ped_x += 1.2 * DT + _gauss(0, 0.05)
			elif self.ped_intent == PedIntent.HESITATING:
				if (self.vehicle_speed < 2.0 or dist > 15.0) and self._rng.random() < 0.4:
					self.ped_intent = PedIntent.CROSSING
					self.ped_vx = 1.2
				elif self._rng.random() < 0.1:
					self.ped_intent = PedIntent.RETREATING
					self.ped_vx = -1.2
			elif self.ped_intent == PedIntent.RETREATING:
				self.ped_x += self.ped_vx * DT + _gauss(0, 0.05)
				if self.ped_x <= -0.5:
					self.ped_intent = PedIntent.HESITATING
					self.ped_vx = 0.0

		# Clamp ped_x
		self.ped_x = max(-0.5, min(self.ped_x, ROAD_WIDTH + 0.5))

		# Collision check
		ped_in_road = 0.0 <= self.ped_x <= ROAD_WIDTH
		dist_to_crossing = abs(self.vehicle_x - CROSSING_Y)
		self.collision = (
			ped_in_road and
			abs(self.ped_y - CROSSING_Y) < 1.0 and
			abs(self.vehicle_x - CROSSING_Y) < 1.5 and
			abs(self.ped_x - ROAD_WIDTH/2) < 2.0 and
			self.vehicle_speed > 0.5
		)
		self.crossed = self.ped_x >= ROAD_WIDTH + 0.5
		self.vehicle_crossed = self.vehicle_x > CROSSING_Y + 5.0

		# Belief update
		moved_into_road = (prev_ped_x < 0.0 and self.ped_x >= 0.0)
		likelihoods = {
			PedIntent.CROSSING: 0.8 if moved_into_road else 0.1,
			PedIntent.HESITATING: 0.3 if moved_into_road else 0.6,
			PedIntent.RETREATING: 0.05 if moved_into_road else 0.8,
		}
		for k in self._belief:
			self._belief[k] *= likelihoods[k]
		norm = sum(self._belief.values()) or 1e-9
		for k in self._belief:
			self._belief[k] /= norm

		# Reward calculation
		safety = -20.0 if self.collision else 0.0
		if not self.collision and dist_to_crossing < 8.0 and ped_in_road:
			safety += -1.0 * (self.vehicle_speed / 8.0) * (1 - min(dist_to_crossing, 8.0)/8.0)
		efficiency = self.vehicle_speed * 0.02
		if self.vehicle_crossed and not self.collision:
			efficiency += 5.0
		smoothness = -0.05 * abs(accel - self.prev_accel)
		belief_accuracy = 0.1 * self._belief[self.ped_intent]
		total = safety + efficiency + smoothness + belief_accuracy
		reward = Reward(
			total=total,
			safety=safety,
			efficiency=efficiency,
			smoothness=smoothness,
			belief_accuracy=belief_accuracy,
		)
		self.prev_accel = accel

		# Log step
		log_entry = {
			"step": self.step_count,
			"action": action.value,
			"reward": float(total),
			"collision": self.collision,
			"vehicle_x": self.vehicle_x,
			"speed": self.vehicle_speed,
			"ped_x": self.ped_x,
			"intent": self.ped_intent.value,
		}
		self._episode_log.append(log_entry)

		# Termination
		self.step_count += 1
		# BUG 1: Only end when ped fully retreats home, not just crossed
		ped_retreated_home = (
			self.ped_intent.value == "RETREATING" and self.ped_x <= -0.4
		)
		if (
			self.collision or
			self.vehicle_crossed or
			ped_retreated_home or
			self.step_count >= MAX_STEPS
		):
			self.done = True

		obs = self._make_obs()
		return obs, reward, self.done, {}

	def state(self) -> dict:
		return {
			"vehicle_x": self.vehicle_x,
			"vehicle_speed": self.vehicle_speed,
			"ped_x": self.ped_x,
			"ped_y": self.ped_y,
			"ped_vx": self.ped_vx,
			"step": self.step_count,
			"collision": self.collision,
			"crossed": self.crossed,
			"vehicle_crossed": self.vehicle_crossed,
			"true_intent": self.ped_intent.value,
			"belief": {k.value: v for k, v in self._belief.items()},
			"_episode_log": self._episode_log,
		}

	@property
	def episode_log(self):
		return self._episode_log
