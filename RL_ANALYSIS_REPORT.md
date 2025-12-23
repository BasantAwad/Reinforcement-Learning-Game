# Reinforcement Learning Algorithms Comparison Report
**Generated:** 2025-12-23 23:44:05

---

## 1. Game Environment Overview

### JungleDash Environment

**Description:**
JungleDash is a grid-based navigation game where an agent (Pink Monster) must navigate through a jungle environment to reach a treasure while collecting coins and avoiding obstacles and traps.

**Environment Specifications:**
- **Grid Size:** 8x8 (64 states)
- **Action Space:** Discrete(4) - Up, Down, Left, Right
- **Observation Space:** Discrete(64) - Agent position on grid
- **Max Steps per Episode:** 128 (grid_size² × 2)

**Environment Elements:**
- **Agent (Pink Monster):** Starting position (0, 0)
- **Goal (Treasure):** Located at (7, 7), +100 reward
- **Rewards (Coins):** 4 coins, +20 reward each
- **Obstacles (Rocks):** 6 obstacles, -1 penalty for collision
- **Traps (Pits):** 2 traps, -20 penalty and episode termination

**Reward Structure:**
- Reaching goal: +100 points
- Collecting coin: +20 points
- Hitting obstacle: -1 point (no movement)
- Falling into trap: -20 points (episode ends)
- Timeout (max steps): -5 points
- Normal movement: 0 points

**Environment Modes:**
1. **Static Mode:**
   - Fixed layout (seed=42) - same obstacle/reward positions every episode
   - Rewards persist on map (can be collected multiple times)
   - Goal always gives exactly 100 points
   - Ideal for fair algorithm comparison

2. **Dynamic Mode:**
   - Random layout - different positions each episode
   - Rewards disappear after collection
   - Goal gives 100 + (collected_coins × 20) bonus
   - Tests generalization capability

---

## 2. Reinforcement Learning Algorithms

### 2.1 Dynamic Programming (Value Iteration)

**Type:** Model-Based, Offline Planning

**Algorithm:**
```
Initialize V(s) = 0 for all states
Repeat until convergence:
    For each state s:
        V(s) = max_a Σ p(s',r|s,a)[r + γV(s')]
Extract policy: π(s) = argmax_a Σ p(s',r|s,a)[r + γV(s')]
```

**Key Properties:**
- Requires complete knowledge of environment dynamics (transition probabilities)
- Computes optimal policy before any interaction
- Guaranteed convergence to optimal policy
- Deterministic policy with epsilon-greedy exploration (ε=0.05) to prevent getting stuck

**Hyperparameters:**
- Discount factor (γ): 0.99
- Convergence threshold (θ): 1e-8
- Exploration rate (ε): 0.05

### 2.2 Q-Learning

**Type:** Model-Free, Off-Policy, Temporal Difference

**Algorithm:**
```
Initialize Q(s,a) arbitrarily
For each episode:
    For each step:
        Choose action a using ε-greedy policy
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
```

**Key Properties:**
- Learns optimal policy while following ε-greedy exploration policy
- Off-policy: learns from max Q-value regardless of action taken
- No environment model required
- Converges to optimal Q* with probability 1

**Hyperparameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Exploration rate (ε): 0.1

### 2.3 SARSA

**Type:** Model-Free, On-Policy, Temporal Difference

**Algorithm:**
```
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize s, choose a using ε-greedy
    For each step:
        Take action a, observe r, s'
        Choose a' using ε-greedy from s'
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s', a ← a'
```

**Key Properties:**
- On-policy: learns value of policy being followed
- Uses actual next action (a') chosen by ε-greedy policy
- More conservative than Q-Learning
- Name: State-Action-Reward-State-Action

**Hyperparameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Exploration rate (ε): 0.1

### 2.4 REINFORCE

**Type:** Model-Free, Policy Gradient, Monte Carlo

**Algorithm:**
```
Initialize policy parameters θ
For each episode:
    Generate episode using π(a|s,θ)
    For each step t in episode:
        Gt ← return from step t
        θ ← θ + α·Gt·∇log π(at|st,θ)
```

**Key Properties:**
- Policy-based method (not value-based)
- Learns stochastic policy directly
- Monte Carlo: updates after complete episodes
- Can handle high-dimensional/continuous action spaces

**Hyperparameters:**
- Learning rate (α): varies by implementation
- Discount factor (γ): 0.99

---

## 3. Experimental Results and Analysis

### 3.1 DP Performance

**Environment Mode:** STATIC

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Episodes | 101 |
| Average Reward | 2355.55 ± 53.24 |
| Final 10 Avg Reward | 2332.90 |
| Max Reward | 2455.00 |
| Min Reward | 2215.00 |
| Average Steps | 6528.00 ± 3731.81 |
| Final 10 Avg Steps | 12352.00 |
| Success Rate | 0.00% |
| Final 10 Success Rate | 0.00% |
| Total Penalties | 509.00 |
| Average Penalties/Episode | 5.04 |

### 3.2 Q-Learning Performance

**Environment Mode:** STATIC

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Episodes | 127 |
| Average Reward | 1192.87 ± 684.52 |
| Final 10 Avg Reward | 1264.60 |
| Max Reward | 2235.00 |
| Min Reward | -9.00 |
| Average Steps | 8121.70 ± 4645.84 |
| Final 10 Avg Steps | 15566.00 |
| Success Rate | 0.00% |
| Final 10 Success Rate | 0.00% |
| Total Penalties | 197.00 |
| Average Penalties/Episode | 1.55 |

### 3.3 SARSA Performance

**Environment Mode:** STATIC

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Episodes | 100 |
| Average Reward | 978.33 ± 335.23 |
| Final 10 Avg Reward | 1064.90 |
| Max Reward | 1195.00 |
| Min Reward | -9.00 |
| Average Steps | 6464.00 ± 3694.86 |
| Final 10 Avg Steps | 12224.00 |
| Success Rate | 0.00% |
| Final 10 Success Rate | 0.00% |
| Total Penalties | -179.00 |
| Average Penalties/Episode | -1.79 |

### 3.4 REINFORCE Performance

**Environment Mode:** STATIC

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Total Episodes | 101 |
| Average Reward | 105.19 ± 115.12 |
| Final 10 Avg Reward | 116.80 |
| Max Reward | 511.00 |
| Min Reward | -39.00 |
| Average Steps | 4958.58 ± 2824.94 |
| Final 10 Avg Steps | 9221.20 |
| Success Rate | 5.94% |
| Final 10 Success Rate | 10.00% |
| Total Penalties | 1449.00 |
| Average Penalties/Episode | 14.35 |

### Overall Best Performer

**Algorithm:** DP

**Final Average Reward:** 2332.90

---

## 4. Challenges Faced and Solutions Implemented

### Challenge 1: Unfair Algorithm Comparison in Dynamic Environments

**Problem:**
- Initial implementation had a dynamic environment where rewards disappeared after collection
- This caused inconsistent evaluation across algorithms
- Dynamic Programming computed policy based on initial state but environment changed during execution
- Different episodes had different reward distributions, making performance comparison unreliable

**Solution:**
- Implemented toggleable environment modes: Static and Dynamic
- Static mode uses fixed seed (42) for consistent layout across episodes
- Rewards persist in static mode, maintaining environment consistency
- Goal reward is fixed at 100 in static mode (no dynamic bonus)
- All algorithms now tested on identical task for fair comparison

**Impact:**
- Fair and reproducible algorithm comparison
- DP algorithm works correctly with consistent transition model
- Performance metrics are directly comparable

### Challenge 2: DP Agent Getting Stuck on Obstacles

**Problem:**
- Dynamic Programming computes deterministic policy offline
- If policy leads agent to obstacle, agent stays in same state
- Policy never updates (computed once), creating infinite loops

**Solution:**
- Added epsilon-greedy exploration (ε=0.05) to DP agent
- Agent occasionally takes random actions to escape bad situations
- Maintains mostly optimal behavior while preventing stuck states

### Challenge 3: Real-time Performance Monitoring

**Problem:**
- Needed to track exploration vs exploitation behavior
- Required comprehensive metrics for analysis and comparison

**Solution:**
- Implemented epsilon tracking displayed in real-time GUI
- Built comprehensive metrics aggregation system
- Tracks: rewards, steps, success rate, penalties, exploration rate
- Automated report generation for analysis

---

## 5. Raw Experimental Data

```json
{
  "algorithms": {
    "DP": [
      {
        "run_id": "DP_static_20251223_232518",
        "env_mode": "static",
        "summary": {
          "total_episodes": 101,
          "avg_reward": 2355.5544554455446,
          "std_reward": 53.23708884475048,
          "max_reward": 2455.0,
          "min_reward": 2215.0,
          "final_avg_reward": 2332.9,
          "avg_steps": 6528.0,
          "std_steps": 3731.8092127009922,
          "final_avg_steps": 12352.0,
          "success_rate": 0.0,
          "final_success_rate": 0.0,
          "total_penalties": 509.0,
          "avg_penalties": 5.03960396039604
        }
      }
    ],
    "Q-Learning": [
      {
        "run_id": "Q-Learning_static_20251223_232946",
        "env_mode": "static",
        "summary": {
          "total_episodes": 127,
          "avg_reward": 1192.8661417322835,
          "std_reward": 684.5153393449262,
          "max_reward": 2235.0,
          "min_reward": -9.0,
          "final_avg_reward": 1264.6,
          "avg_steps": 8121.700787401574,
          "std_steps": 4645.837704167458,
          "final_avg_steps": 15566.0,
          "success_rate": 0.0,
          "final_success_rate": 0.0,
          "total_penalties": 197.0,
          "avg_penalties": 1.5511811023622046
        }
      }
    ],
    "SARSA": [
      {
        "run_id": "SARSA_static_20251223_233537",
        "env_mode": "static",
        "summary": {
          "total_episodes": 100,
          "avg_reward": 978.33,
          "std_reward": 335.23400946204725,
          "max_reward": 1195.0,
          "min_reward": -9.0,
          "final_avg_reward": 1064.9,
          "avg_steps": 6464.0,
          "std_steps": 3694.856966108431,
          "final_avg_steps": 12224.0,
          "success_rate": 0.0,
          "final_success_rate": 0.0,
          "total_penalties": -179.0,
          "avg_penalties": -1.79
        }
      }
    ],
    "REINFORCE": [
      {
        "run_id": "REINFORCE_static_20251223_234022",
        "env_mode": "static",
        "summary": {
          "total_episodes": 101,
          "avg_reward": 105.18811881188118,
          "std_reward": 115.11541456607173,
          "max_reward": 511.0,
          "min_reward": -39.0,
          "final_avg_reward": 116.8,
          "avg_steps": 4958.584158415842,
          "std_steps": 2824.9392152133173,
          "final_avg_steps": 9221.2,
          "success_rate": 5.9405940594059405,
          "final_success_rate": 10.0,
          "total_penalties": 1449.0,
          "avg_penalties": 14.346534653465346
        }
      }
    ]
  },
  "best_performer": {
    "algorithm": "DP",
    "run_id": "DP_static_20251223_232518",
    "final_avg_reward": 2332.9
  },
  "comparison_date": "2025-12-23T23:44:05.649767"
}
```

---

**End of Report**
