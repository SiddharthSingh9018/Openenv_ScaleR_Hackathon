
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from environment import PedestrianNegotiationEnv, Action, Observation, Reward
from graders import GRADERS

TASKS = ["task_1_static", "task_2_stochastic", "task_3_adversarial"]

def test_reset_returns_observation():
	env = PedestrianNegotiationEnv("task_1_static")
	obs = env.reset()
	assert isinstance(obs, Observation)
	assert obs.step == 0
	assert obs.vehicle_x == 0.0

def test_step_returns_types():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	obs, reward, done, info = env.step(Action.COAST)
	assert isinstance(obs, Observation)
	assert isinstance(reward, Reward)
	assert isinstance(done, bool)
	assert isinstance(info, dict)

def test_step_counter_increments():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	env.step(Action.COAST)
	assert env.step_count == 1

def test_vehicle_moves_on_accel():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	env.step(Action.STRONG_ACCEL)
	assert env.vehicle_x > 0.0

def test_speed_clamped_zero():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	env.step(Action.STRONG_BRAKE)
	assert env.vehicle_speed >= 0.0

def test_speed_clamped_max():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	for _ in range(20):
		env.step(Action.STRONG_ACCEL)
		if env.done:
			break
	assert env.vehicle_speed <= 15.0

def test_episode_ends_at_max_steps():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	for _ in range(100):
		if env.done:
			break
		env.step(Action.COAST)
	assert env.step_count <= 80

def test_reset_clears_state():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	env.step(Action.COAST)
	env.reset()
	assert env.step_count == 0

def test_step_after_done_raises():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	for _ in range(100):
		if env.done:
			break
		env.step(Action.COAST)
	with pytest.raises(RuntimeError):
		env.step(Action.COAST)

def test_belief_sums_to_one():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	for _ in range(10):
		if env.done:
			break
		obs, _, _, _ = env.step(Action.COAST)
		s = obs.belief_crossing + obs.belief_hesitating + obs.belief_retreating
		assert abs(s - 1.0) < 1e-4

def test_state_contains_keys():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	state = env.state()
	for k in ["vehicle_x", "vehicle_speed", "ped_x", "ped_y", "ped_vx", "step", "collision", "crossed", "vehicle_crossed", "true_intent", "belief", "_episode_log"]:
		assert k in state

@pytest.mark.parametrize("task", TASKS)
def test_task_resets_cleanly(task):
	env = PedestrianNegotiationEnv(task)
	obs = env.reset()
	assert obs.step == 0

@pytest.mark.parametrize("task", TASKS)
def test_task_runs_full_episode(task):
	env = PedestrianNegotiationEnv(task)
	env.reset()
	for _ in range(100):
		if env.done:
			break
		env.step(Action.COAST)
	assert env.done
	assert len(env.episode_log) > 0

def test_task1_pedestrian_static():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	xs = []
	for _ in range(10):
		if env.done:
			break
		env.step(Action.COAST)
		xs.append(env.ped_x)
	assert all(abs(x - xs[0]) < 1e-6 for x in xs)

@pytest.mark.parametrize("task", TASKS)
def test_grader_score_in_range(task):
	env = PedestrianNegotiationEnv(task)
	env.reset()
	for _ in range(100):
		if env.done:
			break
		env.step(Action.COAST)
	fn = GRADERS[task]
	score = fn(env.episode_log)
	assert 0.0 <= score <= 1.0

@pytest.mark.parametrize("task", TASKS)
def test_grader_deterministic(task):
	env = PedestrianNegotiationEnv(task)
	env.reset()
	for _ in range(100):
		if env.done:
			break
		env.step(Action.COAST)
	fn = GRADERS[task]
	score1 = fn(env.episode_log)
	score2 = fn(env.episode_log)
	assert score1 == score2

@pytest.mark.parametrize("task", TASKS)
def test_grader_empty_log_zero(task):
	fn = GRADERS[task]
	assert fn([]) == 0.0

@pytest.mark.parametrize("task", TASKS)
def test_grader_collision_zero(task):
	env = PedestrianNegotiationEnv(task)
	env.reset()
	# Force a collision in log
	log = [{"collision": True, "vehicle_x": 0, "speed": 0, "ped_x": 0, "intent": "CROSSING", "action": "STRONG_BRAKE", "step": 0, "reward": 0}]
	fn = GRADERS[task]
	assert fn(log) == 0.0

def test_graders_dict_all_tasks():
	for t in TASKS:
		assert t in GRADERS

def test_reward_shape():
	env = PedestrianNegotiationEnv("task_1_static")
	env.reset()
	obs, reward, done, info = env.step(Action.COAST)
	assert isinstance(reward.total, float)
	assert isinstance(reward.safety, float)
	assert isinstance(reward.efficiency, float)
	assert isinstance(reward.smoothness, float)
	assert isinstance(reward.belief_accuracy, float)
	assert 0.0 <= reward.belief_accuracy <= 0.11
