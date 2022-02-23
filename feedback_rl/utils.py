import numpy as np
import torch

def traj_to_input(traj, params, xi_initial):
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if params.dim() == 1:
        params = params.unsqueeze(0)
    if xi_initial.dim() == 1:
        xi_initial = xi_initial.unsqueeze(0)
    # Reset pos, vel to relative values
    traj[:, :, :2] -= xi_initial.view(traj.shape[0], 1, -1)
    # Use first state of traj
    net_input = torch.cat((traj[:, 0], params), axis=1).float().cuda()
    return net_input

def obs_to_input(obs, params, xi_initial):
    params = torch.tensor([params], dtype=torch.float)
    obs = obs.copy()
    obs[:2] -= xi_initial
    obs = torch.tensor([obs], dtype=torch.float)
    net_input = torch.cat((obs, params), axis=1)
    return net_input
