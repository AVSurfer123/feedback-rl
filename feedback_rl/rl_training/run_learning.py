"""
@Mohsin

This is the file that will train the model on the passed in parameters 

Configure the model and environment parameters in params.py and then use this to run
"""

import os
import gym
import json
import numpy as np
import torch as th
import datetime
import gym
import gym_cartpole_swingup

from dotmap import DotMap

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from feedback_rl.envs import OfflineEtaEnv

BASE_PATH = os.path.join(os.path.dirname(__file__), "../../runs")

def run_learning(params):

    #------------Setup Folders and log file------------------------------

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%m_%d_%Y_%H_%M_%S") + "_offline_rl"
    path = os.path.join(BASE_PATH, folder_name)

    models_path = os.path.join(path, "models")
    os.makedirs(models_path, mode=0o775)

    tensorboard_log = os.path.join(path, "tensorboard_log")
    os.makedirs(tensorboard_log, mode=0o775)

    #-----------Train the model on the environment---------

    if params.eta_model == "default":
        env = gym.make("CartPoleSwingUp-v0")
        eval_env = gym.make("CartPoleSwingUp-v0")
    else:
        env = OfflineEtaEnv(params.eta_model)
        eval_env = OfflineEtaEnv(params.eta_model)

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
    callbacks = [eval_callback, save_callback]

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
    model.learn(total_timesteps=params.timesteps, callback=callbacks)

    with open(os.path.join(path, "params.json"), 'w') as f:
        json.dump(params.toDict(), f, indent=2)

if __name__=="__main__":
    # TODO turn this into an argument parser
    params = DotMap()
    params.eta_model = "02_23_2022_11_57_19_eta_model" #path to eta model to plug into 
    params.timesteps = 20000
    params.eval_freq = 1000
    params.save_freq = 10000
    params.gamma = 0.98
    params.learning_rate = 0.0003
    params.policy_kwargs = dict(activation_fn=th.nn.Tanh)


    run_learning(params)

