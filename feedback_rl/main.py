"""Trains RL through feedback linearization."""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cartpole_swingup

from feedback_rl.models import MLP, EtaData
from feedback_rl.spline_planner import SplinePath

FPS = 60
g = 9.81 # [m/s^2]
sim_dt = 0.01 # [s] time between each step of the environment
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

def collect_data(args):
    data = []
    env = gym.make("CartPoleSwingUp-v0")
    obs = env.reset()
    steps = 0
    run = 0

    while steps < args.num_steps:
        done = False
        data.append([obs])
        # print("Initial state of run:", obs)
        spline_duration = args.prediction_horizon * sim_dt # [s] Length that spline is valid for
        traj_des = SplinePath(args.num_knots + 1, spline_duration, sim_dt)
        spline_params = traj_des.random_spline(spline_duration / args.num_knots * 2)[1:]
        n_iterations = int(traj_des.T / sim_dt)
        for i in range(n_iterations):
            action = feedback_controller(env, i, traj_des)
            obs, rew, done, info = env.step(action)
            steps += 1
            data[run].append(obs)
            if done:
                break
        obs = env.reset()
        if done:
            continue
        traj_act = np.array(data[run])
        data[run] = (traj_act, spline_params)

        # traj_des.plot(derivs=False)
        # plt.plot(traj_des.times_eval, traj_act[:, 0], label='x actual')
        # plt.plot()
        # plt.legend()
        # plt.savefig('small_horizon.png')
        # plt.show()

        run += 1

    return data


def train(args, loader):
    eta_dynamics = MLP(5 + args.num_knots, 3 * args.prediction_horizon // args.eta_skip, [32, 32]).cuda()
    print(eta_dynamics)
    optimizer = optim.Adam(eta_dynamics.parameters(), lr=args.learning_rate)

    losses = []
    
    for i in range(args.num_epochs):
        total_loss = 0
        for traj, params in loader:
            net_input = torch.cat((traj[:, 0], params), axis=1).float().cuda()
            target = traj[:, 1:, 2:].cuda()
            target = target.reshape(target.shape[0], -1)

            optimizer.zero_grad()
            output = eta_dynamics(net_input)
            loss = F.mse_loss(output, target)
            total_loss = total_loss + loss
            loss.backward()
            optimizer.step()

        print(f"Loss at iteration {i}: {total_loss}")
        losses.append(total_loss)
    
    return eta_dynamics, losses

def main(args):
    start=  time.time()
    data = collect_data(args)
    print(f"Took {time.time() - start} seconds to collect data")
    print("Data length:", len(data))
    # [print(d[0].shape, end=' ') for d in data]
    dataset = EtaData(data, skip=args.eta_skip)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    start = time.time()
    model, losses = train(args, loader)
    print(f"Took {time.time() - start} seconds to train model for {args.num_epochs} epochs")

    plt.plot(losses, 'o-', label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()


def test_controller():
    env = gym.make("CartPoleSwingUp-v0")
    obs = env.reset()
    num_knots = 10
    traj_des = SplinePath(num_knots + 1, 5, sim_dt)
    knots = np.sin(np.arange(num_knots + 1))
    print(knots)
    traj_des.build_splines(knots)
    data = [obs]
    n_iterations = int(traj_des.T / sim_dt)
    for i in range(n_iterations):
        action = feedback_controller(env, i, traj_des)
        print("Action:", action)
        obs, rew, done, info = env.step(action)
        data.append(obs)
        env.render()
        time.sleep(1/FPS)
        if done:
            break
    data = np.array(data)
    traj_des.plot()
    plt.plot(traj_des.times_eval, data[:, 0], label='x actual')
    plt.legend()
    # plt.savefig('tracking_control_test.png')
    plt.show()


def teleop():
    env = gym.make("CartPoleSwingUp-v0")
    env.reset()
    done = False
    import cv2
    cv2.imshow('blank', np.zeros((300, 400, 3)))
    while not done:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 97:
            action = -1
        elif key == 100:
            action = 1
        else:
            print(key)
            action = 0
        obs, rew, done, info = env.step(action)
        print(obs[0])
        env.render()
        time.sleep(1 / 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help="Number of epochs during training")
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="Batch size for training the model")
    parser.add_argument('-lr', '--learning-rate', type=float, default=.001, help="Learning rate for gradient descent")
    parser.add_argument('-s', '--num-steps', type=int, default=10_000, help="Number of environment steps used to train the Eta model")
    parser.add_argument('-ph', '--prediction-horizon', type=int, default=20, help="Number of timesteps the Eta model should predict into the future")
    parser.add_argument('-es', '--eta-skip', type=int, default=1, help="Number of environment steps between each Eta prediction")
    parser.add_argument('-k', '--num-knots', type=int, default=10, help="Number of knot points along spline that are specified by learned model")
    parser.add_argument('--test-controller', action='store_true', help="Whether to test the controller instead")
    args = parser.parse_args()
    
    if args.test_controller:
        test_controller()
        exit()

    main(args)
