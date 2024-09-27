import gym
import numpy as np
from tqdm import trange

# Initialize Environment
env = gym.make('Taxi-v3')
env.action_space.seed(50)
observation, info = env.reset(seed=50)

# Hyperparameters
num_state = 500
num_action = 6
epsilon = 0.001
alpha = 0.7
gamma = 0.6

# Variables
rewards = []
Q = np.zeros((num_state, num_action))

def epsilon_greedy(Q, S):
    if np.random.rand() < epsilon:
        # perfrom random action
        A = env.action_space.sample()
    else:
        #perform best action
        A = np.argmax(Q[S])
    return A

for _ in trange(10000):
    # Initialize S
    S, info = env.reset()

    terminated = truncated = False

    # Total return for recording
    total_reward = 0

    while not (terminated or truncated):
        # Choose A from S using epsilon greedy
        A = epsilon_greedy(Q, S)
        # Take action A, observe R, S'
        S_prime, reward, terminated, truncated, info = env.step(A)
        # update Q-table
        Q[S, A] = Q[S, A] + alpha * (reward + gamma * np.max(Q[S_prime, :]) - Q[S, A])
        # update S
        S = S_prime
        # update total_reward
        total_reward += reward
    rewards.append(total_reward)
env.close()

# Save Q-table to file
import torch
torch.save(torch.Tensor(Q), '110062126_hw1_2_taxi_qlearning.pth')