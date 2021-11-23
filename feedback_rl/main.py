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
    steps = 0
    run = 0

    while steps < num_steps:
        env.reset()
        done = False
        data.append([])
        while not done:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            steps += 1
            data[run].append(obs)
        data[run] = np.array(data[run])
        run += 1

    return data
        


def main(args):
    start=  time.time()
    data = collect_data(10_000)
    print(f"Took {time.time() - start} seconds to collect data")
    print(len(data))
    [print(d.shape) for d in data]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(args)
