# 🚦 Traffic Signal Control using Reinforcement Learning (DQN)

An intelligent traffic management system built using **Deep Q-Learning (DQN)** to automatically control signal timing and reduce congestion.  
The agent learns **when to switch signals** by observing real-time vehicle queue lengths at a four-way intersection.  

---

## 🧠 Problem Statement

Traffic congestion at intersections is a major urban challenge.  
Traditional signal systems use fixed timers — they **don’t adapt** to dynamic vehicle flow.  

This project applies **Reinforcement Learning** to allow a traffic signal agent to:
- Learn the **optimal switching policy**
- Minimize **average waiting time**
- Improve **traffic flow efficiency**

---

## 🧩 Project Overview

| Component | Description |
|------------|-------------|
| **Environment** | Custom Gym environment simulating a 4-way intersection |
| **Agent** | SARSA |
| **Reward** | Negative of total waiting time (to encourage faster flow) |
| **Visualization** | Pygame-based intersection display with queue bars |
| **Frameworks** | PyTorch, Gymnasium, Numpy, Matplotlib, Pygame |

---

## 🧰 Tech Stack

- 🧠 **Python 3.10+**
- 🧩 **Gymnasium** — RL environment interface  
- 🔥 **PyTorch** — DQN neural network  
- 📊 **Matplotlib** — Training visualization  
- 🎮 **Pygame** — Traffic signal simulation  
- 🧮 **Numpy**, **tqdm**, **random**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/traffic-signal-rl.git
cd traffic-signal-rl
