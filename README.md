# Imitation Learning in MPC using Differentiable Convex Optimization

Code for Numerical Optimization Project, Albert Ludwigs University of Freiburg.



## How to Use

#### Generate training data
```
python src/gen_train_data_norm.py --n_episodes 25 --length 200 --f_name train_200_norm.pkl
```
#### Training Policies

```
python src/diff_mpc.py --mode sysid  --seed 127
```
Modes: 'mpc' (Differentaible MPC) ,  'sysid' (System identification)
#### Testing Policies

```
python src/test_diff_mpc.py --model_path ./src/data/sysid127/model_state_e_0.pt
```

## Files

diff_mpc.py - Differentaible MPC implementation using CvxpyLayers and Pytorch. Used to get baseline perfomrances.

continous_cartpole_env.py - Carpole Environment with Linearized dynamics

mpc.py - Quad Cost MPC implented in Cvxpy as ideal form of the problem.

data - Folder containing trained policies and terminal log.

