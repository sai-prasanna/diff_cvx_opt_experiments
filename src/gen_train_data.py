from imitation_learning import build_mpc_control_policy, get_model_matrix
import random
import tqdm
from continuous_cartpole_env import CartPoleEnv
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import warnings
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import pickle

if __name__=="__main__":
    random.seed(42)
    env = CartPoleEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=20)
    A, B = get_model_matrix(env)
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([1.0])
    nx = 4
    nu = 1
    T = 25
    control_method = 'mpc'
    policy = build_mpc_control_policy(nx, nu, T, A, B, Q, R, env.tau)

    data=[]
    episode_rewards = []

    for i in tqdm.tqdm(range(1000)):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            nstate, action, cost = policy(state)
            action = np.clip(action, -1.0, 1.0)
            data.append([state, nstate, action, cost])
            state, reward, terminated, truncated, _ = env.step([action])
            episode_reward += reward
            # env.render()
        episode_rewards.append(episode_reward)
    pickle.dump(data, open("train.pkl", "wb"))
    print(np.mean(episode_rewards))
