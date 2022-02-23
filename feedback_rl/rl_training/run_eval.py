import os
import argparse
import time
import json
import gym
import numpy as np
import gym_cartpole_swingup

from dotmap import DotMap
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

from feedback_rl.envs import OfflineEtaEnv

BASE_PATH = os.path.join(os.path.dirname(__file__), "../../runs")

def evaluate(folder_name, model_name="best_model", render=False, iterations=10):
	path = os.path.join(BASE_PATH, folder_name)

	with open(os.path.join(path, "params.json"), 'r') as f:
		params = DotMap(json.load(f))

	models_path = os.path.join(path, "models")
	model = SAC.load(os.path.join(models_path, model_name))

	env_results = dict()

	if params.eta_model == "default":
		env = gym.make("CartPoleSwingUp-v0")
	else:
		env = OfflineEtaEnv(params.eta_model)

	evaluations = np.load(os.path.join(path, "evaluations.npz"))
	env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

	obs = env.reset()
	done = False
	total_reward = 0

	while not done:
		action, _states = model.predict(obs)
		obs, reward, done, info = env.step(action)
		total_reward += reward
		if render:
			env.render()
			time.sleep(1/60)
	env.close()

	print("Episode reward:", total_reward)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder', '-f', type=str, default=None, help='Folder to evaluate.  Default: None')
	parser.add_argument('--iterations', '-i', type=str, default=10, help='Number of iterations')
	args = parser.parse_args()
	if args.folder:
		evaluate(args.folder, render=True)
	else:
		evaluate("02_16_2022_103854_", render=True, iterations=args.iterations)

