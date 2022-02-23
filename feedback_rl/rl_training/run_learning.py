"""
@Mohsin

This is the file that will train the model on the passed in parameters 

Configure the model and environment parameters in params.py and then use this to run
"""

import os
import gym
import numpy as np
import torch as th
import datetime
import dill as pickle # so that we can pickle lambda functions
import gym
import gym_cartpole_swingup

from dotmap import DotMap

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

BASE_PATH = os.path.join(os.path.dirname(__file__), "../../runs")

def run_learning(params):

    #------------Setup Folders and log file------------------------------

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%m_%d_%Y_%H%M%S") + "_"

    path = os.path.join(BASE_PATH, folder_name)
    os.mkdir(path)

    models_path = os.path.join(path, "models")
    os.mkdir(models_path)

    tensorboard_log = os.path.join(path, "tensorboard_log")
    os.mkdir(tensorboard_log)

    #-----------Train the model on the environment---------

    if params.eta_model == "default":
        env = gym.make("CartPoleSwingUp-v0")
        eval_env = gym.make("CartPoleSwingUp-v0")
    else:
        env = Cartpole_env(params.eta_model)
        eval_env = Cartpole_env(params.eta_model)

    #create callback function to occasionally evaluate the performance
    #of the agent throughout training
    eval_callback = EvalCallback(eval_env,
                             best_model_save_path=models_path,
                             n_eval_episodes=5,
                             eval_freq=params.eval_freq,
                             log_path=path,
                             deterministic=True,
                             render=False)

    save_callback = CheckpointCallback(save_freq=params.save_freq, 
                                        save_path=models_path,
                                        name_prefix='rl_model')

    #create list of callbacks that will be chain-called by the learning algorithm
    callback = [eval_callback, save_callback]

    # Make Model
    #command to run tensorboard from command prompt
    model = SAC(MlpPolicy,
                env,
                gamma = params.gamma,
                learning_rate = params.learning_rate,
                use_sde = True,
                policy_kwargs=params.policy_kwargs,
                verbose = 1,
                device="cuda",
                tensorboard_log = tensorboard_log
                )

    # Execute learning   
    model.learn(total_timesteps=params.timesteps, callback=callback)

    with open(os.path.join(path, "params.pkl"), 'wb') as pick:
        pickle.dump(params, pick, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    # TODO turn this into an argument parser
    params = DotMap()
    params.eta_model = "default" #path to eta model to plug into 
    params.timesteps = 20000
    params.eval_freq = 1000
    params.save_freq = 10000
    params.gamma = 0.98
    params.learning_rate = 0.0003
    params.policy_kwargs = dict(activation_fn=th.nn.Tanh)


    run_learning(params)

