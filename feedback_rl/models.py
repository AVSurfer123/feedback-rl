import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

