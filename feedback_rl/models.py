import os
import json

from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from feedback_rl.splines import SPLINE_MAP

class EtaData(Dataset):
    def __init__(self, data, skip=1):
        super().__init__()
        self.data = data
        self.skip = skip

        # Initialize traj array with all horizon-length rollouts from our data
        self.traj = []
        for run, params, xi_init in data:
            # T = len(run)
            # for t in range(T):
                # end = t + horizon
                # if end >= T:
                #     break
                # assert len(run[t:end+1:skip]) == horizon // skip + 1, "Error in logic"
            self.traj.append((run[::skip], params, xi_init))

    def __getitem__(self, idx):
        return self.traj[idx]

    def __len__(self):
        return len(self.traj)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
        model = []
        prev_h = input_size
        for h in hidden_sizes:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.append(nn.Linear(prev_h, output_size))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def from_file(model_name):
        model_dir = os.path.join(os.path.dirname(__file__), '../runs', model_name)
        args_path = os.path.join(model_dir, 'args.json')
        with open(args_path, 'r') as f:
            args = DotMap(json.load(f))
        model_path = os.path.join(model_dir, 'model.pt')
        state_dict = torch.load(model_path)
        model = MLP(5 + SPLINE_MAP[args.spline_type].num_segment_params, 3 * args.prediction_horizon // args.eta_skip, [32, 32])
        model.load_state_dict(state_dict)
        return model, args
