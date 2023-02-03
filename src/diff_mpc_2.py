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
from torch import optim
import control



def build_mpc_control_policy(nx, nu, T, tau):
    x = cp.Variable((nx, T + 1))
    u = cp.Variable((nu, T))
    x_0 = cp.Parameter(nx)
    Q = cp.Parameter(nx,nonneg=True)
    R = cp.Parameter(nu,nonneg=True)
    A = cp.Parameter((nx, nx))
    B = cp.Parameter((nx, nu))

    #A, B = get_model_matrix(m,M,l_bar)
    cost = 0.0
    constr = []


    for t in range(T):
        #cost += cp.quad_form(x[:, t + 1], Q.value)
        #cost += cp.quad_form(u[:, t], R.value)
        #cost += x[:, t+1].H @ Q @ x[:, t+1] 
        #cost += u[:, t].H @ R @ u[:, t]
        cost+=cp.sum(cp.multiply(Q, cp.square(x[:,t+1]))) + cp.multiply(R,cp.sum_squares(u[:,t]))
        constr += [x[:, t + 1] == (x[:, t] + tau * (A @ x[:, t] + B @ u[:, t]))]
        constr += [cp.norm(u[:, t], 'inf') <= 1.0]
    # print(x0)
    constr += [x[:, 0] == x_0]
    prob = cp.Problem(cp.Minimize(cost), constr)
    
    def policy(x0, Q_p, R_p, A_p, B_p):
        x_0.value = x0
        Q.value = Q_p.data.numpy()
        R.value = R_p.data.numpy()
        A.value = A_p
        B.value = B_p
        prob.solve(verbose=False)
        if prob.status != cp.OPTIMAL:
            warnings.warn("Solver did not converge!")
            return 0
        ou = np.array(u.value[0, :]).flatten()
        x_t = np.array(x.value[:, 1]).flatten()
        
        c = cp.sum(cp.multiply(Q, cp.square(x[:,1]))) + cp.multiply(R,cp.sum_squares(u[:,0]))

        return x_t, ou[0], c.value

    return policy

def get_model_matrix(env_params):
    
    g = float(9.8)
    M = env_params[0]
    m = env_params[1]
    l_bar = env_params[2]

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
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1)            
    nx = 4
    nu = 1
    T = 25
    env_params = torch.tensor(
                        ( 3.0, 0.1, 1.0), requires_grad=True) #mass cart , masspole, length
    Q = torch.ones((nx), requires_grad=True)
    R = torch.ones((nu), requires_grad=True)
    A = np.zeros((nx,nx))
    B = np.zeros((nx,nu))
    print(np.diag([1.0]))
    params_list = [env_params, Q, R]
    params = [{
                'params': params_list,
                'lr': 1e-2,
                'alpha': 0.5,
            }]
    opt = optim.RMSprop(params)

    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    state, _ = env.reset()
    print(policy(state,Q,R,A,B))

if __name__ == '__main__':
    main()