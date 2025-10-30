import argparse, os
import torch
import numpy as np
from traffic_env import TrafficEnv
from agent_dqn import QNetwork
from utils import set_seed
from tqdm import trange

def evaluate(args):
    env = TrafficEnv(max_queue=args.max_queue, arrival_rate=args.arrival_rate, seed=args.seed)
    set_seed(args.seed)
    state_dim = 4
    action_dim = 2
    net = QNetwork(state_dim, action_dim)
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net.eval()

    rewards = []
    for ep in trange(args.episodes, desc='Eval'):
        state, _ = env.reset(seed=args.seed)
        ep_reward = 0.0
        for t in range(args.max_steps):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = net(s)
            action = int(q.argmax().item())
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--max_queue', type=int, default=20)
    parser.add_argument('--arrival_rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    evaluate(args)
