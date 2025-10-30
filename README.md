# Traffic Signal Control using DQN

**Project:** Intelligent Traffic Light Control using Deep Q-Network (DQN)
**Author:** (Replace with your name)

## Overview
This repository contains a self-contained simulation and DQN implementation to learn traffic-signal control policies at a single 4-way intersection.

## Contents
- `traffic_env.py` - Custom Gymnasium environment simulating a 4-way intersection.
- `replay_buffer.py` - Experience replay buffer implementation.
- `agent_dqn.py` - DQN agent (PyTorch) with network, select_action, and train functions.
- `train.py` - Training loop: runs episodes, training the agent and saving checkpoints.
- `evaluate.py` - Evaluate a trained model and print metrics; optional visualization.
- `utils.py` - Helper utilities (plots, seed setting).
- `requirements.txt` - Python dependencies.
- `README.md` - This file.

## Quickstart (local)
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train the agent:
   ```bash
   python train.py --episodes 1000 --save-dir models/
   ```
3. Evaluate the agent (after training):
   ```bash
   python evaluate.py --model models/dqn_checkpoint.pt --episodes 50
   ```

## Git / GitHub workflow (local)
```bash
git init
git add .
git commit -m "Initial project scaffold: custom env + DQN agent + training loop"
# Create a repo on GitHub and push:
git remote add origin https://github.com/<your-username>/traffic-rl.git
git branch -M main
git push -u origin main
```
