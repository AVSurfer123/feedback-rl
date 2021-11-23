"""Trains RL through feedback linearization."""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup

from feedback_rl.models import MLP

def collect_data(num_steps):
    data = []
    env = gym.make("CartPoleSwingUp-v0")
    env.reset()
    done = False
    steps = 0

    while steps < num_steps:
        action = env.action_space.sample()
        print(action)
        obs, rew, done, info = env.step(action)
        steps += 1
        
        


def main():
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()