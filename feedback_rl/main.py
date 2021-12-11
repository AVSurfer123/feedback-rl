"""Trains RL through feedback linearization."""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup
from scipy import interpolate

from feedback_rl.models import MLP
from feedback_rl.spline_planner import SplinePath

FPS = 60
g = 9.81 # [m/s^2]
spline_duration = .2 # [s] Length that spline is valid for
num_knots = 11 # Number of knot points along spline that are specified by learned model
sim_dt = 0.01 # [s]
pole_inertia_coefficient = 1

def feedback_controller(env, idx, traj):
    v = 0
    M = env.params.cart.mass
    m = env.params.pole.mass
    l = env.params.pole.length
    theta = env.state.theta
    theta_dot = env.state.theta_dot
    xi = np.array([env.state.x_pos, env.state.x_dot])

    x_des = traj.x_des[idx]
    x_dot_des = traj.dt_x_des[idx]
    xi_des = np.array([x_des, x_dot_des])
    v_ff = traj.ddt_x_des[idx]

    # Choose closed-loop eigenvalues to be -3, -3, using standard CCF dynamics
    K = np.array([-9, -6])
    v = v_ff + K @ (xi - xi_des)

    u_star = -m*l*np.sin(theta) * theta_dot**2  - m*g*np.sin(theta)*np.cos(theta) / (1 + pole_inertia_coefficient)
    F = u_star + (M + m - m*np.cos(theta)**2 / (1 + pole_inertia_coefficient)) * v
    return F


def collect_data(num_steps):
    data = []
    env = gym.make("CartPoleSwingUp-v0")
    env.reset()
    steps = 0
    run = 0

    while steps < num_steps:
        done = False
        data.append([])
        spline_params = np.random.uniform(-10, 10, num_knots)
        traj = SplinePath(spline_params, spline_duration, sim_dt)
        traj.build_splines()
        n_iterations = int(spline_duration / sim_dt)
        for i in range(n_iterations):
            action = feedback_controller(env, i, traj)
            obs, rew, done, info = env.step(action)
            steps += 1
            data[run].append(obs)
            if done:
                env.reset()
                break
        data[run] = np.array(data[run])
        run += 1

    return data


def train(data):
    eta_dynamics = MLP(5, 2, [32, 32]).to('cuda')
    print(eta_dynamics)
    optimizer = optim.Adam(eta_dynamics.parameters(), lr=.001)

    # Implement the k-step prediction dataloader
    
    for traj in data:
        optimizer.zero_grad()
        output = eta_dynamics(state)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        

def main(args):
    start=  time.time()
    data = collect_data(10_000)
    print(f"Took {time.time() - start} seconds to collect data")
    print(len(data))
    [print(d.shape) for d in data]
    train(data)

def test_controller():
    env = gym.make("CartPoleSwingUp-v0")
    env.reset()
    spline_params = np.sin(np.arange(num_knots))
    print(spline_params)
    traj = SplinePath(spline_params, 5, sim_dt)
    traj.build_splines()
    traj.plot_path()
    n_iterations = int(traj.T / sim_dt)
    for i in range(n_iterations):
        action = feedback_controller(env, i, traj)
        print("Action:", action)
        obs, rew, done, info = env.step(action)
        env.render()
        time.sleep(1/FPS)
        if done:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    test_controller()

    # main(args)
