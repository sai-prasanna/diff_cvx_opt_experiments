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
from torch.utils.data import TensorDataset, DataLoader



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
    mode = "mpc"
    learn_cost = True
    save_params = True

    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    
    #params of mpc
    env_params = torch.tensor(
                        ( 1.0, 0.1, 0.5), requires_grad=True) #mass cart , masspole, length
    Q = torch.eye((nx), requires_grad=True)
    R = torch.eye((nu), requires_grad=True)
    A, B = get_model_matrix(env_params)
    
    #optimizer and params to learn
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


    

    
    #Dataset hparams
    batch_size = 100
    n_data = 1000

    #load data
    train_data = pickle.load(open('train.pkl', 'rb'))
    x0_train = torch.tensor(train_data[0][:n_data])
    x_hat_train = torch.tensor(train_data[1][:n_data])
    u_hat_train = torch.tensor(train_data[2][:n_data])
    cost_train = torch.tensor(train_data[3][:n_data])
    train_data = TensorDataset(x0_train, x_hat_train, u_hat_train, cost_train)
    trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    #repeat params for batch_size
    if batch_size>1:
        Q=Q.repeat(batch_size, 1, 1)
        R=R.repeat(batch_size, 1, 1)
        A=A.repeat(batch_size, 1, 1)
        B=B.repeat(batch_size, 1, 1)
    

    loss_track=[]

    #train loop
    for i,(x0,x_hat,u_hat,cost) in enumerate(trainLoader):
        opt.zero_grad()
        
        if batch_size>1:
            u = policy(x0,Q,R,A,B)[0][:,0,0].unsqueeze(1)
            u = torch.clip(u, -1.0, 1.0)
            x = x0 + env.tau * torch.bmm(A, x0.unsqueeze(2)).squeeze(2) + torch.bmm(B, u.unsqueeze(1)).squeeze(2)
        
        else:
            u = policy(x0,Q,R,A,B)[0][0][0]
            u = torch.clip(u, -1.0, 1.0)
            x = x0 + env.tau * A @ x0 + B @ u.unsqueeze(0)
        
        if mode == "mpc":             
                im_loss = (u-u_hat).pow(2).mean()
                state_loss = (x-x_hat).pow(2).mean()
                loss=im_loss
        
        elif mode == "sysid":
            state_loss = (x-x_hat).pow(2).mean()
            im_loss = (u.detach()-u_hat).pow(2).mean()
            loss=state_loss
            
        loss.backward()
        opt.step()
        A.detach_()
        B.detach_()
        loss_track.append([im_loss.data, state_loss.data])
    print(loss_track[-1])
    if save_params:
        torch.save(loss_track, 'loss_track.pt')
        torch.save(Q, 'Q.pt')
        torch.save(R, 'R.pt')
        torch.save(A, 'A.pt')
        torch.save(B, 'B.pt')
if __name__ == '__main__':
    main()