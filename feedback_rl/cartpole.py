import gym
import gym_cartpole_swingup
import time

FPS = 60

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
env.reset()
done = False

while not done:
    action = env.action_space.sample()
    print(action)
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(1/FPS)

