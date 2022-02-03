from scipy import interpolate
import numpy as np

from feedback_rl.splines.spline import Spline

class BSpline(Spline):
    
    def __init__(self, num_knots):
        super().__init__(num_knots)
        self.num_knots = num_knots
        self.T = T

        # self.dt = T / (self.shape[1] - 1)
        #array of times to evaluate the paths at
        # self.times = np.arange(0, T + self.dt, self.dt)
        # self.times_eval = np.arange(0, T + self.dt_eval, self.dt_eval)

    
    def build_spline(self, times, points):
        super().build_spline(times, points)
        #build BSpline representation of x path wrt to time 
        self._params = interpolate.splrep([0] + list(times), [0] + list(points))

    def random_spline(self, times, limit):
        super().random_spline(times, limit)
        k = 3
        t = np.concatenate([[0] * ((k+1) // 2), times, [self.T] * ((k+1) // 2)])
        idx = (k+3) // 2
        t[idx] = 0
        t[-idx-1] = self.T
        c = np.random.uniform(-limit, limit, size=self.num_knots)
        c[0] = 0
        self._params = (t, c, k)
        self._x = self.eval_spline(times)
        print(self._x)
        return c

    def deriv(self, t, order):
        super().deriv(t, order)
        return interpolate.splev(t, self._params, der=order)

if __name__ == '__main__':
    num_knots = 10
    times = np.arange(1, num_knots + 1)
    data = times * np.sin(times)
    s = BSpline(num_knots)
    s.build_spline(times, data)
    ax = s.plot()
    s.plot(ax, order=1)
    s.plot(ax, order=2, show=True)
    
    c = s.random_spline(times, 3.0)
    print(c)
    ax = s.plot()
    s.plot(ax, order=1)
    s.plot(ax, order=2, show=True)