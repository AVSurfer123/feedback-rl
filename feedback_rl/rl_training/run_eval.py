import os
import argparse
import gym
import numpy as np
import dill as pickle
import gym_cartpole_swingup

from stable_baselines3 import SAC
import matplotlib.pyplot as plt

BASE_PATH = os.path.join(os.path.dirname(__file__), "../../runs")

def evaluate(folder_name, model="best_model", render=False, iterations=10):

	results = dict()

	path = os.path.join(BASE_PATH, folder_name)

	with open(os.path.join(path, "params.pkl"), 'rb') as f:
		params = pickle.load(f)


	models_path = os.path.join(path, "models")

	model = SAC.load(os.path.join(models_path, model))

	env_results = dict()

	if True: #params.eta_model == "default":
		env = gym.make("CartPoleSwingUp-v0")
	else:
		env = Cartpole_env(params.eta_model)

	evaluations = np.load(os.path.join(path, "evaluations.npz"))

	env_results["mean_reward"] = [evaluations["timesteps"], evaluations["results"]]

	actions = []
	thetas = []

	obs = env.reset()
	done = False

	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		if render:
			env.render()

	env.reset()

	env.close()

	return results

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', '-f', type=str, default=None, help='Folder to evaluate.  Default: None')
	parser.add_argument('-iterations', '-i', type=str, default=10, help='Number of iterations')
	args = parser.parse_args()
	if args.folder:
		evaluate(args.folder, render=True)
	else:
		evaluate("02_16_2022_103854_", render=True, iterations=args.iterations)

