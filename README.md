# Reinforcement Learning – Lunar Lander Project

## Overview
This project implements a Deep Q-Learning (DQN) agent to solve the `LunarLander-v3` environment from Gymnasium.  
The goal is to train an autonomous lander to achieve a soft, fuel-efficient touchdown while maintaining stability—reaching an average reward of at least 200 over 100 consecutive episodes.


## Problem Statement
The Lunar Lander task reflects the challenges faced by autonomous decision-making systems such as self-driving cars or robotic control, where agents must balance short-term actions and long-term rewards under uncertainty.

We designed and evaluated a DQN agent that combines:
- Neural-network function approximation  
- Experience replay  
- Soft target updates  
- ε-greedy exploration  

The environment is considered solved once the moving-average reward over 100 episodes exceeds 200.


## Methodology
1. **Neural Network Architecture** – Three fully connected layers (64-64-output) with ReLU activations; Adam optimizer (α = 5 × 10⁻⁴).  
2. **Experience Replay Buffer** – Capacity = 100 000; mini-batch = 100; breaks temporal correlations for stable learning.  
3. **Target Network & Soft Updates** – θ′ ← τθ + (1 – τ)θ′, τ = 10⁻³ to smooth target shifts.  
4. **ε-Greedy Policy** – ε decays from 1.0 → 0.01 (× 0.995 per episode) to balance exploration and exploitation.  
5. **Training Loop** – Up to 2 000 episodes, 1 000 steps each; discount γ = 0.99; update every 4 steps once buffer ≥ 100 samples.


## Results
- **Solved at ≈ Episode 1 150** (average reward > 200).  
- Smooth, stable landings and efficient fuel use.  
- Approximately 1.1 million timesteps of training.  
- Occasional rare crashes due to ε-greedy exploration near ε = 0.01.

| Metric | Description | Result |
|:--|:--|:--|
| Mean Reward (100 ep avg) | Performance threshold | 200 + |
| Episodes to Solve | Convergence speed | ~ 1 150 |
| Mini-batch Size | Experience sample | 100 |
| τ (Soft Update) | Target smoothing | 1 × 10⁻³ |


## Discussion
**Strengths**
- Stable convergence through soft updates  
- Efficient data reuse via replay  
- Simple, modular design  

**Limitations**
- Long training time  
- Inefficient late-stage ε-greedy exploration  
- Occasional random failures  

**Future Improvements**
- Double DQN to reduce overestimation bias  
- Prioritized Experience Replay  
- Dueling Network Architecture  
- Adaptive exploration (Noisy Nets, UCB)



## Conclusion
The Deep Q-Learning agent surpassed the 200-point average reward threshold after roughly 1 150 episodes, confirming that experience replay, soft target updates, and an ε-greedy schedule enable stable and efficient learning.  
Further extensions such as Double DQN, PER, and Dueling Networks could enhance convergence speed and robustness.
