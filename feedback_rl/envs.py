import numpy as np
import gym
from gym import spaces
import gym_cartpole_swingup

from feedback_rl.splines import ConstAccelSpline, SPLINE_MAP
from feedback_rl.models import MLP
from feedback_rl.utils import obs_to_input

class OfflineEtaEnv(gym.Env):

    def __init__(self, model_name):
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

        self.state = None


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

        # Update state
        self.state[0] += x
        self.state[1] += x_dot
        self.state[2:] = next_eta[-1].detach().numpy()

        reward = self.state[2] # cos(theta)
        done = abs(self.state[0]) > self.env.params.x_threshold

        return self.state, reward, done, {}
        

    def reset(self):
        self.state = self.env.reset()

    def render(self, mode):
        return
