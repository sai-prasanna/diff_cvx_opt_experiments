import random
import tqdm
from continuous_cartpole_env import CartPoleEnv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import warnings
import gymnasium as gym
import numpy as np
import time
import torch
import control


def build_lqr_policy(Q, R, A, B):
   
    # observation: state. 4D in this case.
    # x, v, theta, v_theta = observation
    # cost function 
    K, S, E = control.lqr(A,B,Q,R)
    
    def policy(x):
        action = -1*np.dot(K, x)
        return action[0]
    return policy


def build_mpc_control_policy(nx, nu, T, A, B, Q, R, tau):

    x = cp.Variable((nx, T + 1))
    u = cp.Variable((nu, T))
    x_0 = cp.Parameter(nx)
    cost = 0.0
    constr = []
    for t in range(T):
        #cost += cp.quad_form(x[:, t + 1], Q)
        #cost += cp.quad_form(u[:, t], R)
        cost += cp.square(cp.norm(Q@x[:,t+1])) + cp.square(cp.norm(R@u[:,t]))
        constr += [x[:, t + 1] == (x[:, t] + tau * (A @ x[:, t] + B @ u[:, t]))]
        constr += [cp.norm(u[:, t], 'inf') <= 1.0]
    # print(x0)
    constr += [x[:, 0] == x_0]
    prob = cp.Problem(cp.Minimize(cost), constr)

    def policy(x0):
        x_0.value = x0
        prob.solve(verbose=False)
        if prob.status != cp.OPTIMAL:
            warnings.warn("Solver did not converge!")
            return 0
        ou = np.array(u.value[0, :]).flatten()
        x_t = np.array(x.value[:, 1]).flatten()
        
        c = (cp.quad_form(x[:, 1],Q)+cp.quad_form(u[:, 0],R)).value

        return x_t, ou[0], c

    return policy


def get_model_matrix(env):
    
    m = float(env.masspole)
    M = float(env.masscart)
    l_bar = float(env.length)
    g = float(env.gravity)

    #m = 1.0
    #M = 1.0
    #l_bar = 0.8
    #g = 10.0
    
    
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
    A, B = get_model_matrix(env)
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([1.0])
    nx = 4
    nu = 1
    T = 25
    control_method = 'mpc'
    if control_method == 'lqr':
        policy = build_lqr_policy(Q, R, A, B)
    else:
        policy = build_mpc_control_policy(nx, nu, T, A, B, Q, R, env.tau)

    episode_rewards = []
    for i in tqdm.tqdm(range(10)):
        episode_reward = 0
        state, _ = env.reset()
        #state = np.array([-0.00756797, -0.00285175,  0.03924649,  0.03471813])
        #print(state)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _,action,_ = policy(state)
            #action = np.clip(action, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step([action])
            episode_reward += reward
            # env.render()
        episode_rewards.append(episode_reward)
    print(f":{np.mean(episode_rewards)}")


if __name__ == '__main__':
    main()