"""Base class for splines used in this project."""

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import matplotlib.pyplot as plt

class Spline(ABC):

    def __init__(self, num_knots):
        """
        Constructs the spline with a predefined number of knot points.
        There will be 1 less segment than the number of knot points.

        Args:
            num_knots: number of knot points, i.e. number of points that change their spline segment
        """
        self.num_knots = num_knots
        self.T = None
        self._params = None
        self._t = None
        self._x = None

    @abstractmethod
    def build_spline(self, times, points):
        """
        Uses the provided knot points in the x and y arrays to create spline.
        Must set the self._params variable to the spline parameters.
        """
        assert len(times) == self.num_knots
        self.T = max(times)
        self._t = times
        assert len(points) == self.num_knots
        self._x = points
    
    @abstractmethod
    def random_spline(self, times, limit):
        """
        Generates random spline parameters in self._params.
        
        Returns:
            values at the knot points        
        """
        assert len(times) == self.num_knots
        self.T = max(times)
        self._t = times

    @abstractmethod
    def deriv(self, t, order):
        """
        Gets the derivatives of the spline with respect to time.
        
        Args:
            t: time to evaluate the derivative at
            order: order of the derivative. 0 corresponds to standard function evaluation

        Returns:
            the derivative or function value
        """
        if t < 0 or t > self.T:
            raise ValueError(f"Provided time {t} is out of bounds for the spline: [0, {self.T}]")

    @abstractproperty
    def num_segment_params(self):
        """Number of spline params per segment of the spline."""
        return 0

    def eval_spline(self, times, order=0):
        """Evaluates spline at the given times for the specified amount of derivatives."""
        return np.array([self.deriv(t, order) for t in times])

    def plot(self, ax=None, show=False, order=0, end_time=None):
        """
        Plots the spline at its order'th derivative (0 is the original curve).

        Args:
            ax: matplotlib.pyplot.Axis to use. If None, uses the gca()
            show: whether to show the plot
            order: what derivative of the spline to plot. Default is 0 which is just the function
            end_time: last time to plot the spline to

        Returns:
            matplotlib.pyplot.Axis used for plotting
        """
        if ax is None:
            fig, ax = plt.subplots()

        if self._params is not None:
            if end_time is None:
                end_time = self.T
            times = np.linspace(0, end_time, num=500)
            data = self.eval_spline(times, order)
            if order == 0:
                sim_dt = self._t[1] - self._t[0]
                num_points = int(end_time / sim_dt) + 1
                ax.plot(self._t[:num_points], self._x[:num_points], 'ro', label='$x$ knot points')
            ax.plot(times, data, label=f"$x$ {'' if order == 0 else f'derivative order {order}'}")
            ax.legend()
            ax.grid()
            ax.set(xlabel='Time', ylabel='Value', title='Spline Path')

        if show:
            plt.show()
        return ax

    def x(self, t):
        """Spline value at time `t`."""
        return self.deriv(t, 0)

    @property
    def params(self):
        """Spline parameters"""
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    