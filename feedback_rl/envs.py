import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
import gym_cartpole_swingup
from gym_cartpole_swingup.envs.cartpole_swingup import State

from feedback_rl.splines import ConstAccelSpline, SPLINE_MAP
from feedback_rl.models import MLP
from feedback_rl.utils import obs_to_input

class OfflineEtaEnv(gym.Env):

    def __init__(self, model_name, max_steps=20):
        super().__init__()
        self.env = gym.make("CartPoleSwingUp-v0")
        self.sim_dt = self.env.params.deltat

        # Load args and model
        self.model, self.args = MLP.from_file(model_name)

        # Setup action and observation spaces
        high = np.array([self.args.random_size], dtype=np.float32)
        self.action_space = spaces.Box(low=-high, high=high)
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high)

        # 5-tuple of (x, xdot, cos theta, sin theta, theta dot), obs of underlying env
        self.state = None
        
        self.max_steps = max_steps
        self.step_num = 0


    def step(self, action):
        if self.state is None:
            raise RuntimeError("Must call reset first before step")

        # Get next xi from spline
        spline = ConstAccelSpline(num_knots=2)
        end_time = self.args.prediction_horizon * self.sim_dt
        spline.random_spline([0, end_time], 1)
        spline.params = [action[0]]
        x = spline.deriv(end_time, 0)
        x_dot = spline.deriv(end_time, 1)

        # Get next eta from dynamics model
        net_input = obs_to_input(self.state, spline.params, self.state[:2])
        next_eta = self.model(net_input)[0].reshape(self.args.prediction_horizon, -1)

        # Update state, splines are relative, DNN is absolute
        self.state[0] += x
        self.state[1] += x_dot
        self.state[2:] = next_eta[-1].detach().numpy()

        self.step_num += 1

        reward = self.state[2] # cos(theta)
        done = abs(self.state[0]) > self.env.params.x_threshold or self.step_num == self.max_steps

        return self.state.copy(), reward, done, {}
        

    def reset(self):
        self.state = self.env.reset()
        self.step_num = 0
        return self.state

    def render(self):
        theta = np.sqrt(self.state[2]**2 + self.state[3]**2)
        # self.env.reset()
        self.env.state = State(self.state[0], self.state[1], theta, self.state[4])
        print(self.env.state)
        return self.env.render()

register(
    id="OfflineEta-v0",
    entry_point="feedback_rl.envs:OfflineEtaEnv",
    max_episode_steps=500,
)


if __name__ == '__main__':
    env = OfflineEtaEnv('03_03_2022_16_44_00_eta_model')
    env.reset()
    done = False
    import time
    data = []
    while not done:
        action = env.action_space.sample()
        # print("Action:", action)
        obs, rew, done, info = env.step(action)
        data.append(obs)
        env.render()
        # time.sleep(1/60)
        print(obs, rew, done)

    data = np.array(data)
    import matplotlib.pyplot as plt
    idx = np.arange(len(data))
    plt.figure()
    plt.plot(idx, data[:, 0], label='x')
    plt.figure()
    plt.plot(idx, data[:, 1], label='x dot')
    plt.figure()
    plt.plot(idx, data[:, 2], label='cos')
    plt.figure()
    plt.plot(idx, data[:, 3], label='sin')
    plt.figure()
    plt.plot(idx, data[:, 4], label='theta dot')
    plt.legend()
    plt.show()

    # import time
    # env = gym.make("CartPoleSwingUp-v0")
    # env.reset()
    # print(env.state)
    # for x in np.arange(-2, 2, .1):
    #     print(x)
    #     env.state = State(x, 0, 0, 0)
    #     # print(env.state)
    #     # env.step(1)
    #     # env.reset()
    #     env.render()
    #     time.sleep(.1)

    input()