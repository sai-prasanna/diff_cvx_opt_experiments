import random
import torch
from torch import optim
import tqdm
from continuous_cartpole_env import CartPoleEnv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import warnings
import gymnasium as gym
import numpy as np
import time
import pickle
import control



def build_mpc_control_policy(nx, nu, T, tau):
    x = cp.Variable((nx, T + 1))
    u = cp.Variable((nu, T))
    x_0 = cp.Parameter(nx)
    Q = cp.Parameter((nx,nx),nonneg=True)
    R = cp.Parameter((nu,nu),nonneg=True)
    A = cp.Parameter((nx, nx))
    B = cp.Parameter((nx, nu))

    #A, B = get_model_matrix(m,M,l_bar)
    cost = 0.0
    constr = []


    for t in range(T):
        #cost += cp.quad_form(x[:, t + 1], Q)
        #cost += cp.quad_form(u[:, t], R)
        #cost += x[:, t+1].H @ Q @ x[:, t+1] 
        #cost += u[:, t].H @ R @ u[:, t]
        cost += cp.norm(Q@x[:,t+1]) + cp.norm(R@u[:,t])

        #cost+=cp.sum(cp.multiply(Q, cp.square(x[:,t+1]))) + cp.sum(cp.multiply(R,cp.square(u[:,t])))
        constr += [x[:, t + 1] == (x[:, t] + tau * (A @ x[:, t] + B @ u[:, t]))]
        constr += [cp.norm(u[:, t], 'inf') <= 1.0]
    # print(x0)
    constr += [x[:, 0] == x_0]
    prob = cp.Problem(cp.Minimize(cost), constr)
    return CvxpyLayer(prob, parameters=[x_0,Q,R,A,B], variables=[u])

def get_model_matrix(env):
    
    m = float(env.masspole)
    M = float(env.masscart)
    l_bar = float(env.length)
    g = float(env.gravity)

    #m = 1.0
    #M = 1.0
    #l_bar = 0.6
    #g = 9.8
    
    
    # Model Parameter
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [-1.0 / (l_bar * M)]
    ])

    return A, B

def main():
    random.seed(42)
    env = CartPoleEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)            
    nx = 4
    nu = 1
    T = 25
    
    Q = torch.eye((nx), requires_grad=False)
    R = torch.eye((nu), requires_grad=False)
    A, B = get_model_matrix(env)
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)

    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    episode_rewards = []
    data = [[],[],[],[]]
    for i in tqdm.tqdm(range(100)):
        episode_reward = 0
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            state = torch.tensor(state, dtype=torch.float32)
            u = policy(state,Q,R,A,B)[0][0][0]
            nstate = state + env.tau * A @ state + B @ u.unsqueeze(0)
            data[0].append(state.numpy())
            data[1].append(nstate.numpy())
            data[2].append(u.item())
            action = torch.clip(u, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step([action.data.numpy()])
            #data[1].append(state)
            episode_reward += reward
           
            # env.render()
        episode_rewards.append(episode_reward)
    pickle.dump(data, open("train_200_norm.pkl", "wb"))
    print(np.mean(episode_rewards))

if __name__ == '__main__':
    main()