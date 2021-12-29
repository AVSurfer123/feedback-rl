# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:30:17 2020

@author: mkest
"""
from scipy import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class SplinePath():
    
    def __init__(self, x_shape, T, dt_eval):
        if type(x_shape) == int:
            x_shape = (1, x_shape)
        elif len(x_shape) == 1:
            x_shape = (1, x_shape[0])
        elif len(x_shape) > 2:
            raise RuntimeError(f"x_shape of {x_shape} has too many dimensions. Must have 2 or less")
        self.shape = x_shape

        # sim time
        self.dt_eval = dt_eval
        self.T = T
        self.dt = T / (self.shape[1] - 1)

        #array of times to evaluate the paths at
        self.times = np.arange(0, T + self.dt, self.dt)
        self.times_eval = np.arange(0, T + self.dt_eval, self.dt_eval)

        self.x = None
    
    
    def build_splines(self, x):
        self.x = np.atleast_2d(x)
        assert self.x.shape == self.shape, f"Provided data has shape {self.x.shape} which is different than initialized shape {self.shape}"

        #build BSpline representation of x path wrt to time 
        self.x_spline = interpolate.splprep(self.x, u=self.times, s=0)[0]
        self._eval_splines()

    def random_spline(self, limit):
        k = 3
        t = np.concatenate([[0] * ((k+1) // 2), self.times, [self.T] * ((k+1) // 2)])
        idx = (k+3) // 2
        t[idx] = 0
        t[-idx-1] = self.T
        c = np.random.uniform(-limit, limit, size=self.shape)
        c[0][0] = 0
        self.x_spline = (t, c, k)
        self._eval_splines()
        return c[0]

    def _eval_splines(self):
        #evaluate spline and derivative at desired timesteps        
        self.x_des = interpolate.splev(self.times_eval, self.x_spline, der=0)[0]
        self.dt_x_des = interpolate.splev(self.times_eval, self.x_spline, der=1)[0]
        self.ddt_x_des = interpolate.splev(self.times_eval, self.x_spline, der=2)[0]
    
    def init_arrays_sin(self):
        self.x = np.multiply(self.times, np.sin(self.times))

    def plot(self, ax=None, show=False, derivs=False):
        if ax is None:
            fig, ax = plt.subplots()

        if self.x is not None:
            ax.plot(self.times, self.x[0], 'o', label='$x$ points')
        ax.plot(self.times_eval, self.x_des, label='$x$')
        if derivs:
            ax.plot(self.times_eval, self.dt_x_des, label='$\dot{x}$')
            ax.plot(self.times_eval, self.ddt_x_des, label='$\ddot{x}$')
        ax.legend()
        ax.grid()
        ax.set(xlabel='Time', ylabel='Value', title='Spline Path')

        if show:
            plt.show()
        return ax

if __name__ == '__main__':
    num_knots = 11
    data = np.arange(num_knots) * np.sin(np.arange(num_knots))
    s = SplinePath(num_knots, 10, .01)
    s.build_splines(data)
    print(s.x_spline)
    s.plot(show=True, derivs=True)

    s.x = None
    s.random_spline(1.0)
    print(s.x_spline)
    s.plot(show=True, derivs=True)
