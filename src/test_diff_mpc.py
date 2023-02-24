import argparse
import random
from mpc import get_model_matrix
import gymnasium
from train_diff_mpc import build_mpc_control_policy
import tqdm
from continuous_cartpole_env import CartPoleEnv
import gymnasium as gym

import torch
import numpy as np

def test_solution(model_path,seed):

    #env setup
    random.seed(seed)
    env = CartPoleEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    nx = 4
    nu = 1
    T = 25

    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    
    #Load model state
    model_state = torch.load(model_path)
    
    #Unpack model state
    try:
        A = model_state['A'][-1,:,:]
        B = model_state['B'][-1,:,:]
        Q = model_state['Q'][-1,:,:]
        R = model_state['R'][-1,:,:]
    except:
        A = model_state['A']
        B = model_state['B']
        Q = model_state['Q']
        R = model_state['R']

    print(A,B)
    print(get_model_matrix(env))
    print(model_state['env_params'])

    episode_rewards = []

    #test loop
    for i in tqdm.tqdm(range(10)):
        episode_reward = 0
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            u = policy(torch.tensor(state),Q,R,A,B)[0][0][0]
            u = torch.clip(u, -1.0, 1.0).detach()
            state, reward, terminated, truncated, _ = env.step([u])
            episode_reward += reward
            # env.render()
        episode_rewards.append(episode_reward)
    print(f"mean reward:{np.mean(episode_rewards)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_solution(args.model_path,args.seed)