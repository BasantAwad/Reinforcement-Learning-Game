# RL Arena - Reinforcement Learning Algorithms Comparison Platform

A comprehensive web-based platform for training, visualizing, and comparing reinforcement learning algorithms on a custom 2D grid-world game environment.

![RL Arena](https://img.shields.io/badge/RL-Arena-purple) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-cyan)

## 🎮 Features

- **Custom Game Environment**: JungleDash - A grid-based navigation game with obstacles, rewards, and traps
- **4 RL Algorithms**:
  - **Dynamic Programming** (Value Iteration with ε-greedy exploration)
  - **Q-Learning** (Off-Policy TD Control)
  - **SARSA** (On-Policy TD Control)
  - **REINFORCE** (Policy Gradient)
- **Dual Environment Modes**:
  - **Static Mode**: Fixed layout, persistent rewards (ideal for fair algorithm comparison)
  - **Dynamic Mode**: Random layouts, disappearing rewards (tests generalization)
- **Real-time Visualization**: Live game frames and agent actions via WebSocket
- **Comprehensive Metrics**: Rewards, steps, success rate, penalties, exploration rate (ε)
- **Automated Report Generation**: Export detailed markdown reports for analysis
- **Agent Action Logs**: Real-time color-coded logs showing agent decisions and outcomes

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 📁 Project Structure

```
Reinforcement-Learning-Game/
├── backend/                          # Python FastAPI backend
│   ├── agents/                       # RL algorithm implementations
│   │   ├── base_agent.py             # Abstract base agent class
│   │   ├── dynamic_programming.py    # Value Iteration with ε-greedy
│   │   ├── q_learning.py             # Tabular Q-Learning
│   │   ├── sarsa.py                  # Tabular SARSA
│   │   └── reinforce.py              # Policy Gradient (REINFORCE)
│   ├── envs/                         # Game environments
│   │   └── jungle_dash.py            # Custom grid-world game
│   ├── main.py                       # FastAPI WebSocket server + API endpoints
│   ├── training.py                   # Training orchestration and management
│   ├── metrics.py                    # Performance metrics tracking system
│   ├── report_generator.py           # Automated report generation
│   └── requirements.txt              # Python dependencies
├── frontend/                         # React TypeScript frontend
│   ├── src/
│   │   └── App.tsx                   # Main UI application
│   └── package.json                  # Node dependencies
├── 1 Pink_Monster/                   # Sprite assets for JungleDash
├── README.md                         # This file
├── IMPLEMENTATION_SUMMARY.md         # Feature implementation summary
└── RL_ANALYSIS_REPORT.md             # Generated analysis report

```

---

## 🎯 JungleDash Environment

### Overview

A grid-based navigation game where an agent (Pink Monster) navigates through a jungle to reach treasure while collecting coins and avoiding obstacles.

### Specifications

- **Grid Size**: 8×8 (64 discrete states)
- **Action Space**: Discrete(4) - Up, Down, Left, Right
- **Elements**:
  - Agent starting position: (0, 0)
  - Goal (treasure): (7, 7) - +100 reward
  - Coins: 4 coins - +20 each
  - Obstacles: 6 rocks - -1 penalty for collision
  - Traps: 2 pits - -20 penalty and episode termination
  - Max steps: 128 per episode

### Environment Modes

**Static Mode** (Recommended for Fair Comparison):

- Fixed seed (42) ensures identical layout every episode
- Rewards persist on the map (can be collected repeatedly)
- Goal always gives exactly 100 points
- Perfect for comparing algorithm performance

**Dynamic Mode** (Tests Generalization):

- Randomized layout each episode
- Rewards disappear after collection
- Goal gives 100 + (collected_coins × 20) bonus
- More challenging and realistic

---

## 🤖 Algorithms Implemented

### 1. Dynamic Programming (Value Iteration)

- **Type**: Model-based, offline planning
- **Features**: ε-greedy exploration (5%) to prevent getting stuck
- **Hyperparameters**: γ=0.99, θ=1e-8, ε=0.05

### 2. Q-Learning

- **Type**: Model-free, off-policy TD control
- **Features**: Learns optimal policy while exploring
- **Hyperparameters**: α=0.1, γ=0.99, ε=0.1

### 3. SARSA

- **Type**: Model-free, on-policy TD control
- **Features**: More conservative than Q-Learning
- **Hyperparameters**: α=0.1, γ=0.99, ε=0.1

### 4. REINFORCE

- **Type**: Policy gradient, Monte Carlo
- **Features**: Learns stochastic policy directly
- **Hyperparameters**: γ=0.99

---

## 📊 Metrics & Analysis

### Real-time Metrics Tracked

- Episode rewards (mean, std, min, max)
- Steps per episode
- Success rate (% reaching goal)
- Penalties incurred
- Exploration rate (ε) over time
- Convergence metrics (final 10 episodes)

### Report Generation

Click "Generate Report" in the UI to create a comprehensive markdown file containing:

- Environment overview and specifications
- Algorithm explanations and pseudocode
- Experimental results with performance tables
- Challenges faced and solutions implemented
- Raw JSON data for further analysis

### API Endpoints

- `GET /api/metrics` - View all tracked metrics
- `GET /api/report/generate` - Generate analysis report
- `GET /api/report/download` - Download report file
- `GET /api/metrics/export` - Export metrics to JSON

---

## 🛠️ Technologies

- **Frontend**: React, TypeScript, TailwindCSS, Recharts, Vite
- **Backend**: Python, FastAPI, WebSockets, Pygame, Gymnasium
- **Visualization**: Real-time WebSocket streaming, interactive charts
- **Analysis**: Automated metrics tracking and report generation

---

## 📝 Course Information

**Course**: AIE322 - Advanced Machine Learning  
**Project**: Final Project - Reinforcement Learning Game

### Requirements Satisfied

✅ Design a 2D game using Python (JungleDash with Pygame)  
✅ Implement multiple RL algorithms from scratch (DP, Q-Learning, SARSA, REINFORCE)  
✅ Provide real-time visualization tools for agent performance  
✅ Compare algorithm performance with comprehensive metrics and analysis  
✅ Generate automated reports for evaluation

---

## 🔍 Key Features Implemented

1. **Static/Dynamic Environment Toggle** - Ensures fair algorithm comparison
2. **Epsilon Tracking** - Monitor exploration vs exploitation in real-time
3. **Comprehensive Metrics System** - Track all relevant performance indicators
4. **Automated Report Generation** - Export results for analysis and reporting
5. **Real-time WebSocket Communication** - Live game visualization and statistics

---

## 📄 License

This project was created for educational purposes as part of the AIE322 course.
