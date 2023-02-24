import argparse
import os
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

    cost = 0.0
    constr = []


    for t in range(T):
        cost += cp.norm(Q@x[:,t+1]) + cp.norm(R@u[:,t])
        constr += [x[:, t + 1] == (x[:, t] + tau * (A @ x[:, t] + B @ u[:, t]))]
        constr += [cp.norm(u[:, t], 'inf') <= 1.0]
    constr += [x[:, 0] == x_0]
    prob = cp.Problem(cp.Minimize(cost), constr)
    return CvxpyLayer(prob, parameters=[x_0,Q,R,A,B], variables=[u])

def get_model_matrix_l(env_params):
    
    #setting model paramter to a wrong value
    l_bar = 0.6

    #Creating A,B matrices
    A = torch.zeros((4, 4), requires_grad=False)
    A[0][1] = 1.0
    A[1][2] = - (env_params[1] * env_params[2]) / env_params[0]
    A[2][3] = 1.0
    A[3][2] = env_params[2] * (env_params[0] + env_params[1]) / (l_bar * env_params[0])

    B = torch.zeros((4, 1), requires_grad=False)
    B[1][0] = 1.0 / env_params[0]
    B[3][0] = -1.0 / (l_bar * env_params[0])

    return A, B

def train(mode,seed,learn_cost):

    #setting seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #env setup
    env = CartPoleEnv()
    nx = 4
    nu = 1
    T = 25
    mode = mode #'mpc' or 'sysid'
    learn_cost = learn_cost
    save_params = True

    #build mpc policy
    policy = build_mpc_control_policy(nx, nu, T, env.tau)
    
    #trainable params of mpc
    noise=np.random.uniform(.1,2.0,3)
    env_params = np.array([1.0,0.1,9.8])*noise
    print(env_params) #mass cart , masspole, gravity
    env_params = torch.tensor(env_params,requires_grad=True)

    #Parameter matrces
    Q = torch.eye((nx), requires_grad=False)
    R = torch.eye((nu), requires_grad=False)
    A, B = get_model_matrix_l(env_params)

    #optimizer and params to learn
    lr = 1e-3
    if mode=="sysid":
        params_list = [env_params]
    if mode=="mpc":
        params_list = [env_params]
        if learn_cost:
            params_list += [Q, R]            
    params = [{
                'params': params_list,
                'lr': lr,
               }]
    
    opt = optim.Adam(params)


    
    #Dataset hparams
    batch_size = 1
    n_data = 5000
    epoch = 1
    n_val = 1000
    
    #load data
    data = pickle.load(open('src/train_200_norm.pkl', 'rb'))

    #train_data
    x0_train = torch.tensor(data[0][:n_data])
    x_hat_train = torch.tensor(data[1][:n_data])
    u_hat_train = torch.tensor(data[2][:n_data])
    train_data = TensorDataset(x0_train, x_hat_train, u_hat_train)
    trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    #val_data
    x0_val = torch.tensor(data[0][n_data:n_data+n_val])
    x_hat_val = torch.tensor(data[1][n_data:n_data+n_val])
    u_hat_val = torch.tensor(data[2][n_data:n_data+n_val])
    val_data = TensorDataset(x0_val, x_hat_val, u_hat_val)
    valLoader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    #repeat params for batch_size
    if batch_size>1:
        Q=Q.repeat(batch_size, 1, 1)
        R=R.repeat(batch_size, 1, 1)
        A=A.repeat(batch_size, 1, 1)
        B=B.repeat(batch_size, 1, 1)
         
    #loss tracking
    loss_track=[]
    im_loss_list = []
    state_loss_list = []
    val_loss_track = []
    val_im_loss_list = []
    val_state_loss_list = []

    run_name=mode+str(seed)
    data_path = './src/data/'+run_name
    os.mkdir(data_path)
    
    #Train loop
    for epo in tqdm.tqdm(range(epoch)):
        for i,(x0,x_hat,u_hat) in tqdm.tqdm(enumerate(trainLoader)):
            opt.zero_grad()    
            if batch_size>1:
                u = policy(x0,Q,R,A,B)[0][:,0,0].unsqueeze(1)
                x = x0 + env.tau * torch.bmm(A, x0.unsqueeze(2)).squeeze(2) + torch.bmm(B, u_hat.unsqueeze(1).unsqueeze(1)).squeeze(2)
            
            else:
                u = policy(x0[0],Q,R,A,B)[0][0][0]
                x = x0[0] + env.tau * A @ x0[0] + B @ u_hat.unsqueeze(0)
            
            #loss
            if mode == "mpc":             
                im_loss = (u-u_hat).pow(2).mean()
                state_loss = (x-x_hat).pow(2).mean()
                loss=im_loss
            
            elif mode == "sysid":
                im_loss = (u-u_hat).pow(2).mean()
                state_loss = (x-x_hat).pow(2).mean()
                loss=state_loss
                
            loss.backward()
            opt.step()

            #reinitiate A,B to prevent backward graph issues
            A,B = get_model_matrix_l(env_params)            
            if batch_size>1:
                A=A.repeat(batch_size, 1, 1)
                B=B.repeat(batch_size, 1, 1)

            loss_track.append(loss.item())
            im_loss_list.append(im_loss.item())
            state_loss_list.append(state_loss.item())
            
            #Validation Loop
            if i%1000==0 or i==len(trainLoader)-1:
                with torch.no_grad():
                    val_loss=0
                    state_val_loss=0
                    im_val_loss=0
                    for j,(x0,x_hat,u_hat) in enumerate(valLoader):
                        if batch_size>1:
                            u = policy(x0,Q,R,A,B)[0][:,0,0].unsqueeze(1)
                            x = x0 + env.tau * torch.bmm(A, x0.unsqueeze(2)).squeeze(2) + torch.bmm(B, u_hat.unsqueeze(1).unsqueeze(1)).squeeze(2)
                        
                        else:
                            u = policy(x0[0],Q,R,A,B)[0][0][0]
                            x = x0[0] + env.tau * A @ x0[0] + B @ u_hat.unsqueeze(0)
                        
                        if mode == "mpc":             
                            im_loss = (u-u_hat).pow(2).mean()
                            state_loss = (x-x_hat).pow(2).mean()
                            loss=im_loss
                        
                        elif mode == "sysid":
                            state_loss = (x-x_hat).pow(2).mean()
                            im_loss = (u-u_hat).pow(2).mean()
                            loss=state_loss

                        val_loss+=loss.item()
                        state_val_loss+=state_loss.item()
                        im_val_loss+=im_loss.item()
                val_loss_track.append(val_loss/len(valLoader))
                val_im_loss_list.append(im_val_loss/len(valLoader))
                val_state_loss_list.append(state_val_loss/len(valLoader))
                print(f"epoch:{epo} batch: {i}, val_loss: {val_loss_track[-1]}, val_im_loss: {val_im_loss_list[-1]}, val_state_loss: {val_state_loss_list[-1]}")

        print(f"epoch: {epo}, val_loss: {val_loss_track[-1]}, val_im_loss: {val_im_loss_list[-1]}, val_state_loss: {val_state_loss_list[-1]}")
        print(env_params)
        if((epo+1)%5==0 and epo<epoch-1):
                model_state={"Q":Q, "R":R, "A":A, "B":B, "env_params":env_params}
                torch.save(model_state, f'{data_path}/model_state_e_{epo}.pt')
        
    #save model
    if save_params:
        model_state={"Q":Q, "R":R, "A":A, "B":B, "env_params":env_params}
        loss_metrics={"loss_track":loss_track, "im_loss_list":im_loss_list, "state_loss_list":state_loss_list, "val_loss_track":val_loss_track, "val_im_loss_list":val_im_loss_list, "val_state_loss_list":val_state_loss_list}
        hparams={"batch_size":batch_size, "lr":lr, "epoch":epoch, "n_data":n_data, "mode":mode, "seed":seed, "learn_cost":learn_cost, "n_val":n_val}
        torch.save(loss_metrics, f'{data_path}/loss_metrics.pt')
        torch.save(hparams, f'{data_path}/hparams.pt')
        torch.save(model_state, f'{data_path}/model_state_e_{epo}.pt')
        print(hparams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="mpc", help='mpc or sysid')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--learn_cost', type=bool, default=False, help='learn cost or not')
    args = parser.parse_args()
    
    train(args.mode,args.seed,args.learn_cost)