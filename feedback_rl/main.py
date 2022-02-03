"""Trains RL through feedback linearization."""

import argparse
from ast import Constant
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
from feedback_rl.splines import BSpline, ConstAccelSpline

FPS = 60
g = 9.81 # [m/s^2]
sim_dt = 0.01 # [s] time between each step of the environment
pole_inertia_coefficient = 1

def feedback_controller(env, t, traj, xi_initial):
    v = 0
    M = env.params.cart.mass
    m = env.params.pole.mass
    l = env.params.pole.length
    theta = env.state.theta
    theta_dot = env.state.theta_dot
    xi = np.array([env.state.x_pos, env.state.x_dot])

    x_des = traj.deriv(t, 0)
    x_dot_des = traj.deriv(t, 1)
    xi_des = np.array([x_des, x_dot_des]) + xi_initial # Adding xi_initial since we assume spline value is an offset from starting position
    v_ff = traj.deriv(t, 2)

    # Choose closed-loop eigenvalues to be -3, -3, using standard CCF dynamics
    K = np.array([-900, -60])
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
        traj_des = ConstAccelSpline(args.num_knots)
        num_knots = int()
        spline_params = traj_des.random_spline(spline_duration / args.num_knots * 2)[1:]
        n_iterations = int(traj_des.T / sim_dt)
        xi_initial = np.array([env.state.x_pos, env.state.x_dot])
        for i in range(n_iterations):
            action = feedback_controller(env, i, traj_des, xi_initial)
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
        run += 1

        # traj_des.plot(derivs=False)
        # plt.plot(traj_des.times_eval, traj_act[:, 0], label='x actual')
        # plt.plot()
        # plt.legend()
        # plt.savefig('small_horizon.png')
        # plt.show()

    return data


def train(args, num_params, train_loader, eval_loader):
    eta_dynamics = MLP(5 + num_params, 3 * args.prediction_horizon // args.eta_skip, [32, 32]).cuda()
    print(eta_dynamics)
    optimizer = optim.Adam(eta_dynamics.parameters(), lr=args.learning_rate)

    train_losses = []
    eval_losses = [eval_loss(eta_dynamics, eval_loader)]
    
    for i in range(1, args.num_epochs + 1):
        total_loss = 0
        count = 0
        for traj, params in train_loader:
            net_input = torch.cat((traj[:, 0], params), axis=1).float().cuda()
            target = traj[:, 1:, 2:].cuda()
            target = target.reshape(target.shape[0], -1)

            optimizer.zero_grad()
            output = eta_dynamics(net_input)
            loss = F.mse_loss(output, target)
            total_loss = total_loss + loss
            count += 1

            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / count
        print(f"Training loss at iteration {i}: {epoch_loss}")
        train_losses.append(epoch_loss)

        if i % args.eval_period == 0:
            loss = eval_loss(eta_dynamics, eval_loader)
            print(f"Eval loss at iteration{i}: {loss}")
            eval_losses.append(loss)
    
    return eta_dynamics, train_losses, eval_losses

def eval_loss(model, eval_loader):
    model.eval()
    total_loss = 0
    count = 0
    for traj, params in eval_loader:
        net_input = torch.cat((traj[:, 0], params), axis=1).float().cuda()
        target = traj[:, 1:, 2:].cuda()
        target = target.reshape(target.shape[0], -1)

        output = model(net_input)
        loss = F.mse_loss(output, target)
        total_loss = total_loss + loss
        count += 1
    model.train()
    return total_loss / count

def test_loss(model):
    env = gym.make("CartPoleSwingUp-v0")
    obs = env.reset()
    steps = 0
    run = 0
    done = False
    losses = []

    for i in range(args.test_size):
        # print("Initial state of run:", obs)
        spline_duration = args.prediction_horizon * sim_dt # [s] Length that spline is valid for
        traj_des = BSpline(args.num_knots + 1, spline_duration, sim_dt)
        spline_params = traj_des.random_spline(spline_duration / args.num_knots * 2)[1:]
        xi_initial = np.array([env.state.x_pos, env.state.x_dot])
        while not done:
            action = feedback_controller(env, i, traj_des, xi_initial)
            obs, rew, done, info = env.step(action)
            steps += 1
            
        obs = env.reset()
        if done:
            continue
        traj_act = np.array(data[run])
        data[run] = (traj_act, spline_params)
        run += 1


def main(args):
    start=  time.time()
    data = collect_data(args)
    print(f"Took {time.time() - start} seconds to collect data")
    print("Data length:", len(data))
    # [print(d[0].shape, end=' ') for d in data]
    end_idx = int(len(data) * args.validation_split / 100)
    train_dataset = EtaData(data[:end_idx], skip=args.eta_skip)
    eval_dataset = EtaData(data[end_idx:], skip=args.eta_skip)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    start = time.time()
    model, train_losses, eval_losses = train(args, train_loader, eval_loader)
    print(f"Took {time.time() - start} seconds to train model for {args.num_epochs} epochs")

    plt.plot(train_losses, 'o-', label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    plt.plot(eval_losses, 'o-', label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('eval_loss.png')
    plt.show()


def test_controller():
    env = gym.make("CartPoleSwingUp-v0")
    obs = env.reset()
    num_knots = 10
    traj_des = ConstAccelSpline(num_knots)
    times = np.arange(1, num_knots + 1)
    knots = np.sin(times)
    traj_des.build_spline(times, knots)
    data = [obs]
    xi_initial = np.array([env.state.x_pos, env.state.x_dot])
    t = 0
    while t <= traj_des.T:
        action = feedback_controller(env, t, traj_des, xi_initial)
        print("Action:", action)
        obs, rew, done, info = env.step(action)
        data.append(obs)
        env.render()
        time.sleep(1/FPS)
        t += sim_dt
        if done:
            break
    data = np.array(data)
    
    ax = plt.gca()
    traj_des.plot(ax, order=0)
    traj_des.plot(ax, order=1)
    traj_des.plot(ax, order=2)
    eval_times = np.arange(0, t + sim_dt, sim_dt)
    plt.plot(eval_times, data[:, 0] - xi_initial[0], label='x actual')
    plt.plot(eval_times, data[:, 1] - xi_initial[1], label='x dot actual')
    plt.legend()
    plt.savefig('tracking_sine_offset.png')
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
    parser.add_argument('-ep', '--eval-period', type=int, default=20, help="How many epochs to wait between each eval")
    parser.add_argument('-vp', '--validation-split', type=float, default=20, help="What percentage of collected data to use for eval")
    parser.add_argument('-t', '--test-size', type=int, default=1000, help="How many environments to test on")
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
