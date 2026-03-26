import sys
sys.path.insert(0, 'pedestrian-negotiation-env/app')
from environment import PedestrianNegotiationEnv, Action
sys.path.insert(0, 'pedestrian-negotiation-env/app')
from graders import GRADERS

def rule_based(obs):
    ped_in_road = 0.0 < obs.ped_x < 6.0
    dist        = abs(obs.vehicle_x - obs.ped_y)
    ped_safe    = obs.ped_x <= -0.2 or obs.ped_x >= 6.4
    if ped_in_road and dist < 12.0:   return Action.STRONG_BRAKE
    elif ped_in_road and dist < 22.0: return Action.SOFT_BRAKE
    elif ped_safe:                    return Action.STRONG_ACCEL
    else:                             return Action.COAST

print("Task scores after fix:")
for task in PedestrianNegotiationEnv.TASKS:
    env = PedestrianNegotiationEnv(task, seed=42)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(rule_based(obs))
    score      = GRADERS[task](env._episode_log)
    steps      = len(env._episode_log)
    collisions = sum(1 for s in env._episode_log if s['collision'])
    print(f"  {task}: score={score:.4f}  steps={steps}  collisions={collisions}")

print()
print("Expected:")
print("  task_1_static:      score >= 0.80, collisions = 0")
print("  task_2_stochastic:  score >= 0.35, collisions = 0")
print("  task_3_adversarial: score >= 0.30, collisions = 0")
print()
print("PASS if all scores meet minimums AND collisions = 0")
