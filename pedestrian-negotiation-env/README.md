# pedestrian-negotiation-env

POMDP environment for autonomous vehicle negotiation with a pedestrian crossing an urban road under uncertainty.

- Real-world AV safety scenario
- POMDP with belief-state mechanics
- 3 tasks: static, stochastic, adversarial
- Deterministic, scorable, and hackathon-ready

See `openenv.yaml` for full spec. See `tests/` for test coverage.

## Problem Framing

This environment targets adaptive negotiation for autonomous road crossing under uncertainty. An autonomous vehicle and a pedestrian implicitly negotiate right-of-way over time without explicit communication, while the vehicle must infer pedestrian intent from noisy observations and choose actions that trade off:

- Safety: avoiding collisions at all costs
- Efficiency: minimizing unnecessary delay
- Smoothness: maintaining stable, comfortable control

The underlying research problem is a POMDP with latent pedestrian intent. In this implementation, the control space is a practical approximation of that formulation:

- Hidden state includes vehicle dynamics and pedestrian intent
- Observations expose noisy pedestrian motion and current belief values
- Actions are discrete longitudinal controls: `STRONG_BRAKE`, `SOFT_BRAKE`, `COAST`, `SOFT_ACCEL`, `STRONG_ACCEL`
- Rewards combine safety, efficiency, smoothness, and belief accuracy

The full final problem statement is captured in [PROBLEM_STATEMENT.md](/c:/Users/sidme/OneDrive/Desktop/MetaHck/pedestrian-negotiation-env/PROBLEM_STATEMENT.md).

## Abstraction Boundary

This environment intentionally isolates longitudinal right-of-way negotiation at a crosswalk, allowing clean evaluation of safety, intent inference, and sequential decision-making without the confounds of full 2D urban driving.

It is not a full urban driving simulator. The benchmark is designed to stress the decision problem that matters here:

- when to yield
- when to proceed
- how to react to uncertain, changing pedestrian intent

The 1D formulation is a deliberate abstraction for controlled evaluation, reproducibility, and clearer grading.

## Rubric Notes

- Real-world utility: models safety-critical AV-pedestrian right-of-way negotiation with noisy, partially observed human intent.
- Task quality: includes three tasks with explicit difficulty progression from trivial curbside waiting to adversarial hesitation and re-entry.
- Environment design: seeded stochasticity, belief-state observations, dense reward shaping, and clear episode termination conditions.
- Spec compliance: FastAPI endpoints, typed Pydantic models, Docker support, tests, and deterministic graders.
- Creativity: frames road-crossing as sequential negotiation under uncertainty rather than simple obstacle avoidance.

## Tasks

- `task_1_static`: sanity-check task for efficient and smooth crossing when the pedestrian stays at the curb.
- `task_2_stochastic`: medium task where the pedestrian may cross, pause, or retreat, requiring intent inference.
- `task_3_adversarial`: hard task with hesitation, retreat, and re-entry behavior that forces repeated micro-negotiation under uncertainty and punishes brittle or overly reactive policies.

## Grading

- Scores are deterministic and bounded in `[0.0, 1.0]`.
- Any collision produces a score of `0.0`.
- `task_1_static` emphasizes safe, smooth, efficient traversal.
- `task_2_stochastic` adds intent-aware action quality and efficiency.
- `task_3_adversarial` emphasizes robustness under adversarial uncertainty, intent-responsive control, and penalties for oscillatory brake-accelerate behavior.
- The grader is intentionally stricter on the hard task: safe completion alone is not enough if the policy is indecisive or repeatedly over-corrects.

## Baseline Scores (Latest Results)

| Task                | Score   | Steps | Collisions |
|---------------------|---------|-------|------------|
| task_1_static       | 0.9600  | 7     | 0          |
| task_2_stochastic   | 0.7850  | 30    | 0          |
| task_3_adversarial  | 0.6930  | 30    | 0          |

All baseline requirements are met: scores exceed minimums and collisions are zero for all tasks.

## Verification

- `pytest pedestrian-negotiation-env/tests/test_environment.py`
- `python -m py_compile pedestrian-negotiation-env/app/environment.py pedestrian-negotiation-env/app/main.py pedestrian-negotiation-env/baseline/run_baseline.py`
