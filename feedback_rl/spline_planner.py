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
    
    def __init__(self, x_coord, T, dt_eval):
        self.x = np.atleast_2d(x_coord)

        # sim time
        self.dt_eval = dt_eval
        self.T = T
        self.dt = T / (len(x_coord) - 1)

        #array of times to evaluate the paths at
        self.times = np.arange(0, T + self.dt, self.dt)
        self.times_eval = np.arange(0, T + self.dt_eval, self.dt_eval)

        self.x_des = np.zeros(len(self.times_eval))
    
    
    def build_splines(self):
        #build BSpline representation of x path wrt to time 
        x_spline = interpolate.splprep(self.x, u=self.times, s=0)[0]
        #evaluate spline and derivative at desired timesteps        
        self.evaluate_splines(x_spline)
        
    def evaluate_splines(self, x_spline):
        self.x_spline = x_spline

        self.x_des = interpolate.splev(self.times_eval, x_spline, der=0)[0]
        self.dt_x_des = interpolate.splev(self.times_eval, x_spline, der=1)[0]
        self.ddt_x_des = interpolate.splev(self.times_eval, x_spline, der=2)[0]

        print(self.x_des, self.dt_x_des, self.ddt_x_des)

    
    def init_arrays_sin(self):
        self.x = np.multiply(self.times, np.sin(self.times))

    def plot_path(self):
        fig, ax = plt.subplots()
        ax.plot(self.times, self.x[0], 'o', label='x points')
        ax.plot(self.times_eval, self.x_des, label='x')
        
        #ax.set(xlim=(-10, 10), ylim=(-10, 10))
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.legend()
        ax.set(xlabel='Time', ylabel='X-coordinate', title='Desired Path vs Learned path')
        ax.grid()
        #deffig.savefig("test.png")
        plt.show()

if __name__ == '__main__':
    data = np.arange(11) * np.sin(np.arange(11))
    s = SplinePath(data, 10, .01)
    s.build_splines()
    s.plot_path()
