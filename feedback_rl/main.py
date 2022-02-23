"""Trains an eta dynamics model through feedback linearization."""

import argparse
import time
import datetime
import os
import json

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
from feedback_rl.splines import SPLINE_MAP
from feedback_rl.utils import traj_to_input, obs_to_input

DEBUG = False
BASE_PATH = os.path.join(os.path.dirname(__file__), "../runs")

FPS = 60
g = 9.81 # [m/s^2]
sim_dt = 0.01 # [s] time between each step of the environment, defined in gym_cartpole_swingup/envs/cartpole_swingup.py
pole_inertia_coefficient = 1
max_steps = 500 # Defined in gym_cartpole_swingup/__init__.py

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
        data.append([[obs]])
        spline_num = 0
        # spline_duration = args.prediction_horizon * sim_dt # [s] Length of spline segment
        num_knots = max_steps // args.prediction_horizon + 1
        traj_des = SPLINE_MAP[args.spline_type](num_knots)
        times = np.arange(0, max_steps + 1, args.prediction_horizon) * sim_dt
        spline_knots = traj_des.random_spline(times, args.random_size)
        xi_initial = np.array([env.state.x_pos, env.state.x_dot])
        for i in range(max_steps):
            action = feedback_controller(env, i * sim_dt, traj_des, xi_initial)
            obs, rew, done, info = env.step(action)
            steps += 1
            data[run][spline_num].append(obs)
            if done:
                break
            if (i + 1) % args.prediction_horizon == 0:
                data[run].append([obs])
                spline_num += 1
        obs = env.reset()
        if (i + 1) % args.prediction_horizon != 0:
            del data[run][spline_num]
            spline_num -= 1
            if len(data[run]) == 0:
                del data[run]
                continue

        for num in range(spline_num + 1):
            traj_act = np.array(data[run][num])
            data[run][num] = (traj_act, np.array([traj_des.params[num]]), xi_initial)
            if DEBUG:
                eval_times = np.arange(args.prediction_horizon * num, args.prediction_horizon * (num + 1) + 1) * sim_dt
                plt.plot(eval_times, traj_act[:, 0] - xi_initial[0], label='x actual')
        
        run += 1

        if DEBUG:
            traj_des.plot(ax=plt.gca(), end_time=None)
            plt.legend()
            plt.savefig('small_horizon.png')
            plt.show()

    return data

def train_model(args, train_loader, eval_loader):
    eta_dynamics = MLP(5 + SPLINE_MAP[args.spline_type].num_segment_params, 3 * args.prediction_horizon // args.eta_skip, [32, 32]).cuda()
    print(eta_dynamics)
    optimizer = optim.Adam(eta_dynamics.parameters(), lr=args.learning_rate)

    train_losses = []
    eval_losses = []
    
    for i in range(1, args.num_epochs + 1):
        total_loss = 0
        count = 0
        for traj, params, xi_initial in train_loader:
            net_input = traj_to_input(traj, params, xi_initial)
            # Targets are eta variables after 1st timestep
            target = traj[:, 1:, 2:].cuda()
            target = target.reshape(target.shape[0], -1)

            optimizer.zero_grad()
            output = eta_dynamics(net_input)
            loss = F.mse_loss(output, target, reduction='sum')
            total_loss = total_loss + loss
            count += traj.shape[0]

            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / count
        print(f"Training loss at iteration {i}: {epoch_loss}")
        train_losses.append(epoch_loss)

        if i % args.eval_period == 0:
            loss = eval_loss(eta_dynamics, eval_loader)
            print(f"Eval loss at iteration {i}: {loss}")
            eval_losses.append(loss)
    
    return eta_dynamics, train_losses, eval_losses

def eval_loss(model, eval_loader):
    model.eval()
    total_loss = 0
    count = 0
    for traj, params, xi_initial in eval_loader:
        net_input = traj_to_input(traj, params, xi_initial)
        # Targets are eta variables after 1st timestep
        target = traj[:, 1:, 2:].cuda()
        target = target.reshape(target.shape[0], -1)

        output = model(net_input)
        loss = F.mse_loss(output, target, reduction='sum')
        total_loss = total_loss + loss
        count += traj.shape[0]
    model.train()
    return total_loss / count

def test_model(args, eta_model):
    env = gym.make("CartPoleSwingUp-v0")
    done = False
    losses = []

    for _ in range(args.test_size):
        obs = env.reset()
        old_obs = obs
        done = False
        num_knots = max_steps // args.prediction_horizon + 1
        traj_des = SPLINE_MAP[args.spline_type](num_knots)
        times = np.arange(0, max_steps + 1, args.prediction_horizon) * sim_dt
        spline_knots = traj_des.random_spline(times, args.random_size)
        xi_initial = np.array([env.state.x_pos, env.state.x_dot])
        eta_traj = []
        for j in range(max_steps):
            action = feedback_controller(env, j * sim_dt, traj_des, xi_initial)
            obs, rew, done, info = env.step(action)
            eta_traj.append(obs[2:])

            if (j + 1) % args.prediction_horizon == 0:
                net_input = obs_to_input(old_obs, [traj_des.params[(j // args.prediction_horizon) - 1]], xi_initial).cuda()
                target = eta_model(net_input).reshape(args.prediction_horizon, -1).cpu()
                eta_loss = F.mse_loss(torch.tensor(eta_traj), target)
                losses.append(eta_loss)
                old_obs = obs
                eta_traj = []

            if done:
                break
        
    return losses

################################
##### Entrypoint functions #####
################################

def main(args):
    start=  time.time()
    data = collect_data(args)
    print(f"Took {time.time() - start} seconds to collect data")
    
    condensed = []
    for run in range(len(data)):
        for spline_num in range(len(data[run])):
            condensed.append(data[run][spline_num])

    print("Data length:", len(data))
    print("Condensed length:", len(condensed))

    end_idx = int(len(condensed) * args.validation_split / 100)
    train_dataset = EtaData(condensed[end_idx:], skip=args.eta_skip)
    eval_dataset = EtaData(condensed[:end_idx], skip=args.eta_skip)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    start = time.time()
    model, train_losses, eval_losses = train_model(args, train_loader, eval_loader)
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

    test_losses = test_model(args, model)
    print(f"Average test loss from {args.test_size} episodes: {sum(test_losses) / len(test_losses)}")

    if args.save:
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%m_%d_%Y_%H_%M_%S") + "_eta_model"
        folder_name = os.path.join(BASE_PATH, folder_name)
        os.makedirs(folder_name, mode=0o775)
        model_path = os.path.join(folder_name, 'model.pt')
        torch.save(model.state_dict(), model_path)
        args_path = os.path.join(folder_name, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=2)


def test_controller():
    env = gym.make("CartPoleSwingUp-v0")
    obs = env.reset()
    num_knots = 11
    traj_des = SPLINE_MAP[args.spline_type](num_knots)
    times = np.linspace(0, max_steps, num_knots) * sim_dt
    knots = times * np.sin(times) / 4
    traj_des.build_spline(times, knots)
    data = [obs]
    xi_initial = np.array([env.state.x_pos, env.state.x_dot])
    for i in range(max_steps):
        action = feedback_controller(env, i * sim_dt, traj_des, xi_initial)
        # print("Action:", action)
        obs, rew, done, info = env.step(action)
        data.append(obs)
        env.render()
        time.sleep(1/FPS)
        if done:
            break
    data = np.array(data)
    
    ax = plt.gca()
    traj_des.plot(ax, order=0)
    traj_des.plot(ax, order=1)
    traj_des.plot(ax, order=2)
    eval_times = np.arange(0, max_steps + 1) * sim_dt
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
    parser.add_argument('-t', '--test-size', type=int, default=100, help="How many environments to test on")
    parser.add_argument('-s', '--num-steps', type=int, default=100_000, help="Number of environment steps used to train the Eta model")
    parser.add_argument('-ph', '--prediction-horizon', type=int, default=20, help="Number of timesteps the Eta model should predict into the future")
    parser.add_argument('-es', '--eta-skip', type=int, default=1, help="Number of environment steps between each Eta prediction")
    parser.add_argument('-k', '--num-knots', type=int, default=10, help="Number of knot points along spline that are specified by learned model")
    parser.add_argument('-r', '--random-size', type=float, default=2.0, help="Size parameter that is proportional to the randomness when generating splines")
    parser.add_argument('-st', '--spline-type', type=int, default=0, help="Which spline to use for trajectory tracking")
    parser.add_argument('--save', action='store_true', help="Whether to save the trained model")
    parser.add_argument('--load', type=str, help="Model path to load for getting test loss")
    parser.add_argument('--test-controller', action='store_true', help="Whether to test the controller instead")
    parser.add_argument('--teleop', action='store_true', help="Keyboard control of environment")
    args = parser.parse_args()
    
    if args.test_controller:
        test_controller()
    elif args.teleop:
        teleop()
    elif args.load:
        test_size = args.test_size
        model, args = MLP.from_file(args.load)
        args.test_size = test_size
        test_losses = test_model(args, model.cuda())
        print(f"Average test loss from {args.test_size} episodes: {sum(test_losses) / len(test_losses)}")
    else:
        main(args)
