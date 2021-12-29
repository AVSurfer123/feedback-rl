import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class EtaData(Dataset):
    def __init__(self, data, skip=1):
        super().__init__()
        self.data = data
        self.skip = skip

        # Initialize traj array with all horizon-length rollouts from our data
        self.traj = []
        for run, params in data:
            # T = len(run)
            # for t in range(T):
                # end = t + horizon
                # if end >= T:
                #     break
                # assert len(run[t:end+1:skip]) == horizon // skip + 1, "Error in logic"
            self.traj.append((run[::skip], params))

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

