import matplotlib.pyplot as plt
import numpy as np
import random, torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_rewards(rewards, filename=None, window=20):
    plt.figure(figsize=(8,4))
    plt.plot(rewards, label='episode reward')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, window-1+len(smoothed)), smoothed, label=f'smoothed({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
