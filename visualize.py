import pygame
import argparse
import torch
import numpy as np
from traffic_env import TrafficEnv
from agent_dqn import DQNAgent

def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    label = font.render(text, True, color)
    screen.blit(label, (x, y))

def draw_static_intersection(screen, env, action, step, reward, episode, font):
    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (60, 60, 60), (250, 250, 100, 100))
    if action == 0:
        ns_color, ew_color = (0, 255, 0), (255, 0, 0)
    else:
        ns_color, ew_color = (255, 0, 0), (0, 255, 0)
    pygame.draw.circle(screen, ns_color, (300, 220), 10)
    pygame.draw.circle(screen, ns_color, (300, 380), 10)
    pygame.draw.circle(screen, ew_color, (220, 300), 10)
    pygame.draw.circle(screen, ew_color, (380, 300), 10)
    q_north, q_south, q_east, q_west = env.state
    scale = 10
    pygame.draw.rect(screen, (0, 150, 255), (295, 200 - q_north * scale, 10, q_north * scale))
    pygame.draw.rect(screen, (0, 150, 255), (295, 350, 10, q_south * scale))
    pygame.draw.rect(screen, (0, 150, 255), (350, 295, q_east * scale, 10))
    pygame.draw.rect(screen, (0, 150, 255), (200 - q_west * scale, 295, q_west * scale, 10))
    draw_text(screen, f"Episode: {episode}", 10, 10, font)
    draw_text(screen, f"Step: {step}", 10, 40, font)
    draw_text(screen, f"Reward: {reward:.2f}", 10, 70, font)
    draw_text(screen, f"Action: {'N-S Green' if action == 0 else 'E-W Green'}", 10, 100, font)
    pygame.display.flip()

def run_visualization(model_path, episodes=3, max_steps=300, fps=10):
    env = TrafficEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load_model(model_path)
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Traffic RL Visualization - Static Queues")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()
    running = True
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        for step in range(1, max_steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            draw_static_intersection(screen, env, action, step, reward, ep, font)
            state = next_state
            total_reward += reward
            clock.tick(fps)
            if done:
                break
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}")
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Traffic RL Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for visualization")
    args = parser.parse_args()
    run_visualization(args.model, args.episodes, args.max_steps, args.fps)
