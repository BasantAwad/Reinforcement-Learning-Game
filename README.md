# RL Arena - Tabular Reinforcement Learning Platform

A web-based platform for visualizing and training RL agents using tabular methods on classic game environments.

![RL Arena](https://img.shields.io/badge/RL-Arena-purple) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-cyan)

## 🎮 Features

- **4 Game Environments**: Taxi, Blackjack, FrozenLake, Jungle Dash (custom Pygame game)
- **3 RL Algorithms**:
  - **Dynamic Programming** (Value Iteration)
  - **Q-Learning** (Off-Policy TD Control)
  - **SARSA** (On-Policy TD Control)
- **Real-time Visualization**: Live game frames streamed via WebSocket
- **Training Metrics**: Interactive charts showing rewards, penalties, and progress
- **Agent Action Logs**: Real-time color-coded logs showing agent decisions

---

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

---

## 📁 Project Structure

```
Reinforcement-Learning-Game/
├── backend/                  # Python FastAPI backend
│   ├── agents/               # RL AGENT IMPLEMENTATIONS
│   │   ├── base_agent.py     # Base agent class
│   │   └── tabular.py        # Q-Learning, SARSA, DP agents
│   ├── envs/                 # GAME ENVIRONMENTS
│   │   └── jungle_dash.py    # Custom Pygame game (JungleDash)
│   ├── main.py               # FastAPI WebSocket server
│   └── training.py           # Training orchestration
├── frontend/                 # React TypeScript frontend
│   └── src/App.tsx           # Main UI application
├── 1 Pink_Monster/           # Game sprites for JungleDash
└── README.md
```

### Where is the Code?

| Component            | Location                      | Description                           |
| -------------------- | ----------------------------- | ------------------------------------- |
| **Agent Algorithms** | `backend/agents/tabular.py`   | Q-Learning, SARSA, DP implementations |
| **Custom Game**      | `backend/envs/jungle_dash.py` | JungleDash Pygame environment         |
| **Training Loop**    | `backend/training.py`         | WebSocket training orchestration      |
| **UI**               | `frontend/src/App.tsx`        | React game selection & visualization  |

---

## 🎯 Supported Games & Algorithms

| Game            | Type          |  DP  | Q-Learning | SARSA |
| --------------- | ------------- | :--: | :--------: | :---: |
| Taxi            | Gymnasium     |  ✅  |     ✅     |  ✅   |
| Blackjack       | Gymnasium     | ❌\* |     ✅     |  ✅   |
| FrozenLake      | Gymnasium     |  ✅  |     ✅     |  ✅   |
| **Jungle Dash** | Custom Pygame |  ✅  |     ✅     |  ✅   |

\*Blackjack doesn't expose transition probabilities, so DP is not compatible.

---

## 🛠️ Technologies

- **Frontend**: React, TypeScript, TailwindCSS, Recharts, Vite
- **Backend**: Python, FastAPI, WebSockets, Pygame
- **RL**: Gymnasium (Taxi, Blackjack, FrozenLake)

---

## 📝 Course

AIE322 - Advanced Machine Learning Final Project

**Requirements Satisfied**:

- ✅ Design a 2D game using Python (Jungle Dash with Pygame)
- ✅ Implement RL algorithms from scratch (DP, Q-Learning, SARSA)
- ✅ Provide visualization tools for agent performance
- ✅ Compare algorithm performance with metrics and graphs
