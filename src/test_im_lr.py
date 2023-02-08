import argparse
import random
from imitation_learning import get_model_matrix,build_mpc_control_policy
import tqdm
from continuous_cartpole_env import CartPoleEnv
import gymnasium as gym

import torch
import numpy as np

def test_solution(model_path,seed):
    random.seed(seed)
    env = CartPoleEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    nx = 4
    nu = 1
    T = 25

    
    model_state = torch.load(model_path)
    
    try:
        A = model_state['A'][-1,:,:].detach().numpy()
        B = model_state['B'][-1,:,:].detach().numpy()
        Q = model_state['Q'][-1,:,:].detach().numpy()
        R = model_state['R'][-1,:,:].detach().numpy()
    except:
        A = model_state['A'].detach().numpy()
        B = model_state['B'].detach().numpy()
        Q = model_state['Q'].detach().numpy()
        R = model_state['R'].detach().numpy()

    policy = build_mpc_control_policy(nx, nu, T,A,B,Q,R,env.tau)

    print(A,B)
    print(get_model_matrix(env))
    episode_rewards = []
    for i in tqdm.tqdm(range(10)):
        episode_reward = 0
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _,u,_ = policy(state)
            u = np.clip(u, -1.0, 1.0)
            #x = x0 + env.tau * torch.bmm(A, x0.unsqueeze(2)).squeeze(2) + torch.bmm(B, u.unsqueeze(1)).squeeze(2)
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