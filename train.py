import argparse, os, math
import numpy as np
from traffic_env import TrafficEnv
from replay_buffer import ReplayBuffer
from agent_dqn import DQNAgent
from utils import set_seed, plot_rewards
from tqdm import trange

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    env = TrafficEnv(max_queue=args.max_queue, arrival_rate=args.arrival_rate, seed=args.seed)
    set_seed(args.seed)
    buffer = ReplayBuffer(capacity=args.buffer_capacity)
    agent = DQNAgent(state_dim=4, action_dim=2, device=args.device, lr=args.lr,
                     gamma=args.gamma, batch_size=args.batch_size, buffer=buffer,
                     target_update=args.target_update)

    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_decay = args.eps_decay

    rewards = []
    global_step = 0
    for ep in trange(args.episodes, desc='Episodes'):
        state, _ = env.reset(seed=args.seed)
        ep_reward = 0.0
        done = False
        for t in range(args.max_steps):
            epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * global_step / eps_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            state = next_state
            ep_reward += reward
            global_step += 1
        rewards.append(ep_reward)
        # save checkpoint periodically
        if (ep + 1) % args.save_every == 0:
            fname = os.path.join(args.save_dir, f'dqn_checkpoint_ep{ep+1}.pt')
            torch.save(agent.policy_net.state_dict(), fname)
    # final save
    final_path = os.path.join(args.save_dir, 'dqn_checkpoint_final.pt')
    import torch
    torch.save(agent.policy_net.state_dict(), final_path)
    plot_rewards(rewards, filename=os.path.join(args.save_dir, 'rewards.png'))
    print('Training complete. Model saved to', final_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--buffer_capacity', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update', type=int, default=500)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--max_queue', type=int, default=20)
    parser.add_argument('--arrival_rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)
