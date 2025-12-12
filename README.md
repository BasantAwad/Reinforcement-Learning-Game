# RL Arena - Reinforcement Learning Visualization Platform

A web-based platform for visualizing and running RL agents on classic game environments.

![RL Arena](https://img.shields.io/badge/RL-Arena-purple) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-cyan) ![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)

## 🎮 Features

- **Multiple Environments**: MsPacman, KungFuMaster, MiniWorld-Maze
- **RL Algorithms**: Dynamic Programming, Q-Learning, SARSA, DQN, Policy Gradient
- **Real-time Visualization**: Live game frames streamed via WebSocket
- **Training Metrics**: Interactive charts showing rewards and progress

## 🚀 Quick Start

### Backend
```bash
cd c:\Users\Pc\AdvML
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

## 📁 Project Structure

```
AdvML/
├── backend/           # Python FastAPI backend
│   ├── agents/        # RL agent implementations
│   ├── envs/          # Environment wrappers
│   ├── main.py        # FastAPI entry point
│   └── training.py    # Training orchestration
├── frontend/          # React TypeScript frontend
│   └── src/App.tsx    # Main application
└── DOCUMENTATION.md   # Complete documentation
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
- **Backend**: Python, FastAPI, WebSockets, PyTorch
- **RL**: Gymnasium, ALE-py (Atari), MiniWorld

## 📝 Course

AIE322 - Advanced Machine Learning Final Project

---

*Built with ❤️ for reinforcement learning education*
