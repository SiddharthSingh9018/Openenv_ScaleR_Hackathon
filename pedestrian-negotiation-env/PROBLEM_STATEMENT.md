# Adaptive Negotiation for Autonomous Road Crossing under Uncertainty

## Problem Description

An autonomous vehicle operating in an urban environment must interact with a pedestrian attempting to cross the road in the absence of explicit signaling or communication. The vehicle must continuously decide how to control its motion while accounting for uncertainty in the pedestrian's intent and behavior.

The interaction unfolds sequentially over time, where both the vehicle and the pedestrian implicitly negotiate right-of-way through their actions. The pedestrian's behavior is stochastic and only partially observable, influenced by latent intent such as crossing, hesitating, or retreating. The vehicle must infer that intent from observable cues such as pedestrian position and movement.

The objective of the autonomous agent is to learn a policy that maximizes long-term reward by balancing three competing factors:

- Safety: Avoiding collisions at all costs
- Efficiency: Minimizing unnecessary delays in traversal
- Smoothness: Maintaining stable and comfortable control

## Formal Framing

The problem is modeled as a Partially Observable Markov Decision Process (POMDP) defined by:

- State (S): True environment state including vehicle dynamics and pedestrian intent
- Observation (O): Noisy observations of pedestrian position, velocity, and environment features
- Action (A): Vehicle control inputs over time
- Transition Function (P): Evolution of vehicle dynamics and stochastic pedestrian behavior
- Reward Function (R): Penalties for collisions, inefficiency, and unstable control, with rewards for safe and efficient interaction
- Policy (pi): A mapping from observations or belief states to actions that maximizes expected cumulative reward

## Key Challenges

1. Uncertainty in Human Behavior: Pedestrian intent is not directly observable and must be inferred over time.
2. Sequential Decision-Making: Actions influence future states, so long-horizon planning matters.
3. Safety-Efficiency Trade-off: Overly cautious behavior is inefficient, while aggressive behavior increases risk.
4. Partial Observability: The agent must act under incomplete and noisy information.

## Objective

Design and evaluate an RL agent that learns an effective policy for autonomous driving behavior in this setting, demonstrating the ability to:

- Anticipate pedestrian intent
- Adapt actions dynamically over time
- Reduce collision rates while maintaining efficiency
- Converge to stable interaction strategies in repeated scenarios

## Significance

This problem captures a core challenge in real-world autonomous systems: negotiation under uncertainty without explicit communication. It extends classical game-theoretic formulations into a sequential learning setting, enabling more realistic modeling of human-AI interaction in safety-critical environments.

## Implementation Note

The current environment in this repository instantiates the action space as discrete longitudinal controls rather than fully continuous acceleration and steering. This keeps the benchmark simple and hackathon-friendly while preserving the central POMDP structure, latent-intent inference, and safety-efficiency-smoothness trade-off.
