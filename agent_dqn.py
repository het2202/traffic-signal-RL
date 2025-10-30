import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu', lr=1e-3,
                 gamma=0.99, batch_size=64, buffer=None, target_update=100):
        self.device = torch.device(device)
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = buffer
        self.action_dim = action_dim
        self.update_count = 0
        self.target_update = target_update
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return int(qvals.argmax().item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
        expected_q = reward + (1.0 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
