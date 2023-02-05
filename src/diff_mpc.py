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

def get_model_matrix(env_params):
    
    g = float(9.8)


    # Model Parameter
    A=torch.zeros((4,4),requires_grad=False)
    A[0][1]=1.0
    A[1][2]=env_params[1]*g/env_params[0]
    A[2][3]=1.0
    A[3][2]=g*(env_params[0]+env_params[1])/(env_params[2]*env_params[0])

    B = torch.zeros((4,1),requires_grad=False)
    B[1][0]=1.0/env_params[0]
    B[3][0]=-1.0/(env_params[2]*env_params[0])

    return A, B

def main():
    random.seed(42)
    env = CartPoleEnv()
    nx = 4
    nu = 1
    T = 25
    mode = "sysid"
    learn_cost = True

    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    
    env_params = torch.tensor(
                        ( 1.0, 0.1, 0.5), requires_grad=True) #mass cart , masspole, length
    Q = torch.eye((nx), requires_grad=True)
    R = torch.eye((nu), requires_grad=True)
    A, B = get_model_matrix(env_params)
    
    if mode=="sysid":
        params_list = [env_params]
    if mode=="mpc":
        params_list = [env_params]
        if learn_cost:
            params_list += [Q, R]
    params = [{
                'params': params_list,
                'lr': 1e-2,
               }]
    
    opt = optim.Adam(params)


    train_data = pickle.load(open('train.pkl', 'rb'))
    loss_track=[]

    for x0,x_hat,u_hat,cost in train_data[:10]:
        opt.zero_grad()
        x0 = torch.tensor(x0)
        x_hat = torch.tensor(x_hat)
        u_hat = torch.tensor(u_hat)
        u = policy(x0,Q,R,A,B)[0][0][0]
        u = torch.clip(u, -1.0, 1.0)
        if mode == "mpc":
            x = x0 + env.tau * A @ x0 + B @ u.unsqueeze(0)
            im_loss = (u-u_hat).pow(2).mean()
            state_loss = (x-x_hat).pow(2).mean()
            loss=im_loss
        if mode == "sysid":
            x = x0 + env.tau * A @ x0 + B @ u.unsqueeze(0)
            state_loss = (x-x_hat).pow(2).mean()
            im_loss = (u.detach()-u_hat).pow(2).mean()
            loss=state_loss

        loss.backward()
        opt.step()
        A.detach_()
        B.detach_()
        loss_track.append([im_loss.data, state_loss.data])
    print(loss_track[-1])

if __name__ == '__main__':
    main()