# pedestrian-negotiation-env

POMDP environment for autonomous vehicle negotiation with a pedestrian crossing an urban road under uncertainty.

- Real-world AV safety scenario
- POMDP with belief-state mechanics
- 3 tasks: static, stochastic, adversarial
- Deterministic, scorable, and hackathon-ready

See `openenv.yaml` for full spec. See `tests/` for test coverage.

## Baseline Scores (Latest Results)

| Task                | Score   | Steps | Collisions |
|---------------------|---------|-------|------------|
| task_1_static       | 1.0000  | 7     | 0          |
| task_2_stochastic   | 0.7729  | 30    | 0          |
| task_3_adversarial  | 0.8893  | 19    | 0          |

All baseline requirements are met: scores exceed minimums and collisions are zero for all tasks.
