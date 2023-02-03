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
        cost += cp.norm(Q@x[:,t+1] + R@u[:,t])
        #cost+=cp.sum(cp.multiply(Q, cp.square(x[:,t+1]))) + cp.sum(cp.multiply(R,cp.square(u[:,t])))
        constr += [x[:, t + 1] == (x[:, t] + tau * (A @ x[:, t] + B @ u[:, t]))]
        constr += [cp.norm(u[:, t], 'inf') <= 1.0]
    # print(x0)
    constr += [x[:, 0] == x_0]
    prob = cp.Problem(cp.Minimize(cost), constr)
    return CvxpyLayer(prob, parameters=[x_0,Q,R,A,B], variables=[u])

def get_model_matrix(env_params):
    
    g = float(9.8)
    M = env_params[0]
    m = env_params[1]
    l_bar = env_params[2]

    # Model Parameter
    A = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ],requires_grad=True)
    B = torch.tensor([
        [0.0],
        [1.0 / M],
        [0.0],
        [-1.0 / (l_bar * M)]
    ],requires_grad=True)

    return A, B

def main():
    random.seed(42)
    env = CartPoleEnv()
    nx = 4
    nu = 1
    T = 25
    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    mode = "mpc"
    t_env_params = [3.0, 1.0, 0.7]
    env_params = [1.0,0.1,0.5]
    Q = torch.eye((nx), requires_grad=True)
    R = torch.eye((nu), requires_grad=True)
    A, B = get_model_matrix(env_params)
    A_hat, B_hat = get_model_matrix(t_env_params)
    if mode=="sysid":
        params_list = [A, B]
    if mode=="mpc":
        params_list = [Q, R, A, B]
    params = [{
                'params': params_list,
                'lr': 1e-2,
               }]
    
    opt = optim.Adam(params)

    """state, _ = env.reset()
    state = torch.tensor([-0.04164756,  0.0237744,   0.04489669, -0.01397501])
    u_hat = torch.tensor(0.4061829582598139)
    #print(Q,R,A,B)
    u=policy(torch.tensor(state),Q,R,A,B)[0][0][0]
    u = torch.clip(u, -1.0, 1.0)
    im_loss = (u-u_hat).pow(2).mean()
    im_loss.backward()
    opt.step()
    print(Q,R,A,B,env_params)
    #print(np.diag(Q.data.numpy())@state.data.numpy())"""

    train_data = pickle.load(open('train.pkl', 'rb'))
    print(A,B)
    loss=0
    for x0,x_hat,u_hat,cost in train_data[:1000]:
        x0 = torch.tensor(x0)
        x_hat = torch.tensor(x_hat)
        u_hat = torch.tensor(u_hat)
        u=policy(x0,Q,R,A,B)[0][0][0]
        u = torch.clip(u, -1.0, 1.0)
        if mode == "mpc":
            im_loss=(u-u_hat).pow(2).mean()
        if mode == "sysid":
            x = x0 + env.tau * A @ x0 + B @ u.unsqueeze(0)
            im_loss = (x-x_hat).pow(2).mean()
        im_loss.backward()
        opt.step()
        loss+=im_loss.data
        #print(im_loss.data)
    print(A,B)
    print(A_hat,B_hat)
    print(loss/1000)
if __name__ == '__main__':
    main()