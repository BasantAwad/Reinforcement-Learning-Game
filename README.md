# RL Arena - Reinforcement Learning Visualization Platform

A web-based platform for visualizing and running RL agents on classic game environments.

![RL Arena](https://img.shields.io/badge/RL-Arena-purple) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-cyan) ![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)

## 🎮 Features

- **Multiple Environments**: 
  - **Tabular**: Taxi, Blackjack, FrozenLake, **Jungle Dash** (custom Pygame)
  - **Visual**: MsPacman, KungFuMaster, MiniWorld-Maze
- **RL Algorithms**: 
  - Dynamic Programming (Value Iteration)
  - Q-Learning (Off-Policy TD)
  - SARSA (On-Policy TD)
  - DQN (Deep Q-Network)
  - Policy Gradient (Actor-Critic)
- **Real-time Visualization**: Live game frames streamed via WebSocket
- **Training Metrics**: Interactive charts showing rewards, penalties, and progress
- **Agent Action Logs**: Real-time color-coded logs showing agent decisions
- **Custom Game**: Jungle Dash - 2D grid game with sprite-based graphics

## 🚀 Quick Start

### Backend
```bash
cd Reinforcement-Learning-Game
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## 🎯 Supported Games & Algorithms

| Game | Type | DP | Q-Learning | SARSA | DQN | PG |
|------|------|:--:|:----------:|:-----:|:---:|:--:|
| Taxi | Tabular | ✅ | ✅ | ✅ | ✅ | ✅ |
| Blackjack | Tabular | ❌ | ✅ | ✅ | ✅ | ✅ |
| FrozenLake | Tabular | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Jungle Dash** | Tabular | ✅ | ✅ | ✅ | ✅ | ✅ |
| MsPacman | Visual | ❌ | ❌ | ❌ | ✅ | ✅ |
| KungFuMaster | Visual | ❌ | ❌ | ❌ | ✅ | ✅ |
| MiniWorld-Maze | Visual | ❌ | ❌ | ❌ | ✅ | ✅ |

## 📁 Project Structure

```
Reinforcement-Learning-Game/
├── backend/              # Python FastAPI backend
│   ├── agents/           # RL agent implementations
│   │   ├── dqn.py        # Deep Q-Network
│   │   ├── policy_gradient.py  # Actor-Critic
│   │   └── tabular.py    # Q-Learning, SARSA, DP
│   ├── envs/             # Environment wrappers
│   │   ├── wrappers.py   # Frame preprocessing
│   │   └── jungle_dash.py # Custom game
│   ├── main.py           # FastAPI entry point
│   └── training.py       # Training orchestration
├── frontend/             # React TypeScript frontend
│   └── src/App.tsx       # Main application
├── 1 Pink_Monster/       # Game sprites
│   ├── Pink_Monster.png  # Agent sprite
│   ├── Rock1.png         # Obstacle sprite
│   └── Rock2.png         # Obstacle sprite
└── DOCUMENTATION.md      # Complete documentation
```

## 📖 Documentation

See [DOCUMENTATION.md](./DOCUMENTATION.md) for comprehensive documentation including:
- Architecture overview
- Class-level documentation
- File-level documentation
- Requirement mapping
- Developer instructions

## 🛠️ Technologies

- **Frontend**: React, TypeScript, TailwindCSS, Recharts, Vite
- **Backend**: Python, FastAPI, WebSockets, PyTorch, Pygame
- **RL**: Gymnasium, ALE-py (Atari), MiniWorld

## 📝 Course

AIE322 - Advanced Machine Learning Final Project

**Project Requirements Satisfied**:
- ✅ Design a 2D game using Python (Jungle Dash with Pygame)
- ✅ Implement RL algorithms from scratch (DP, Q-Learning, SARSA, DQN, PG)
- ✅ Provide visualization tools for agent performance
- ✅ Compare algorithm performance with metrics and graphs

---

*Built with ❤️ for reinforcement learning education*
