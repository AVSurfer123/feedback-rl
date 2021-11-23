# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:30:17 2020

@author: mkest
"""
from scipy import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Spline_Path():
    
    def __init__(self, T, dt, dt_factor=1, x_coord=[0], y_coord=[0]):
        
        #unload salient parameters     
        #final sim time
        self.dt_factor= dt_factor
        self.dt_split = dt/self.dt_factor
        self.T=T
        self.dt = dt
        #array of times to evaluate the paths at
        self.times = np.arange(0, T, dt)
        self.times = self.times.reshape((len(self.times)),1)
        
        self.times_split = np.arange(0, T, self.dt_split)
        self.times_split = self.times_split.reshape((len(self.times_split)),1)

        self.x_des = np.zeros((len(self.times_split),1))
        self.y_des = np.zeros((len(self.times_split),1))
        #array of coefficients to be used for constructing path
        self.x = x_coord
        self.y = y_coord
        #print(x_coord)
        #create test set of points to base x_spline on
    def build_splines(self):
        
        #build BSpline representation of x path wrt to time 
        x_spline=interpolate.splrep(self.times, self.x)
        #evaluate spline and derivative at desired timesteps        
        ##build BSpline representation of y path wrt to time
        y_spline=interpolate.splrep(self.times, self.y)
        #eva;uate spline and derivative at desired timesteps
    
        self.evaluate_splines(x_spline, y_spline)
        
    def evaluate_splines(self, x_spline, y_spline):
        
        self.x_spline=x_spline
        self.y_spline=y_spline

        self.x_des=interpolate.splev(self.times_split, x_spline, der=0)
        self.dt_x_des=interpolate.splev(self.times_split, x_spline, der=1)
        self.ddt_x_des=interpolate.splev(self.times_split, x_spline, der=2)

        self.y_des=interpolate.splev(self.times_split, y_spline, der=0)
        self.dt_y_des=interpolate.splev(self.times_split, y_spline, der=1)
        self.ddt_y_des=interpolate.splev(self.times_split, y_spline, der=2)
    
    def spline_arrays_sin(self):
           
        self.x = np.multiply(self.times,np.sin(self.times))
        self.y = np.multiply(self.times,np.cos(self.times))

    def plot_path(self):
        deffig, ax = plt.subplots()

        ax.plot(self.times_split, self.x_des.reshape((len(self.times_split),1)), label='x')
        ax.plot(self.times_split, self.y_des.reshape((len(self.times_split),1)), label='y')
        
        #ax.set(xlim=(-10, 10), ylim=(-10, 10))
         #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.legend()
        ax.set(xlabel='x-coordinate', ylabel='y-coordinate',
        title='Desired Path vs Learned path')
        ax.grid()
            #deffig.savefig("test.png")
        plt.show()
            
                
    def return_paths(self):
            
        self.x_des = self.x_des.reshape((1,len(self.x_des)))
        self.dt_x_des = self.dt_x_des.reshape((1,len(self.dt_x_des)))
        self.ddt_x_des = self.ddt_x_des.reshape((1,len(self.ddt_x_des)))
        self.y_des = self.y_des.reshape((1,len(self.y_des)))
        self.dt_y_des = self.dt_y_des.reshape((1,len(self.dt_y_des)))
        self.ddt_y_des = self.ddt_y_des.reshape((1,len(self.ddt_y_des)))

        self.state_des=np.concatenate((self.x_des, self.dt_x_des, self.ddt_x_des, \
                                  self.y_des, self.dt_y_des, self.ddt_y_des), axis=0)
        #return arrays providing x,y, derivatives for all time steps. 
        return self.state_des, self.x_spline, self.y_spline
