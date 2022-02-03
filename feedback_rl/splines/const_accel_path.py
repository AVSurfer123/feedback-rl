import numpy as np

from feedback_rl.splines.spline import Spline

class ConstAccelSpline(Spline):

    def __init__(self, num_knots, init_vel=0):
        super().__init__(num_knots)
        self._params = [] # Contains constant acceleration between each knot point
        self.init_vel = init_vel
        self.knot_vels = []

    def build_spline(self, times, points):
        super().build_spline(times, points)
        prev_x = 0
        prev_t = 0
        init_vel = self.init_vel
        self._params = []
        self.knot_vels = []
        for t, x in zip(times, points):
            delta = t - prev_t
            # Inverting x(t) = x(0) + v(0)*t + .5*a*t^2
            a = (x - prev_x - init_vel * delta) * 2 / delta / delta
            self._params.append(a)
            init_vel += a * delta
            self.knot_vels.append(init_vel)
            prev_t = t
            prev_x = x 

    def random_spline(self, times, limit):
        super().random_spline(times, limit)

        self._params = np.random.uniform(low=-limit, high=limit, size=self.num_knots)
        prev_x = 0
        prev_t = 0
        init_vel = self.init_vel
        self._x = []
        self.knot_vels = []
        for t, a in zip(times, self._params):
            delta = t - prev_t
            x = prev_x + init_vel * delta + .5 * a * delta * delta
            self._x.append(x)
            init_vel += a * delta
            self.knot_vels.append(init_vel)
            prev_t = t
            prev_x = x
        
        return self._x

    def deriv(self, t, order):
        super().deriv(t, order)
        if order > 2:
            raise ValueError(f"ConstAccelSpline can only have up to 2nd derivatives, not {order} derivatives")
        
        if order == 0:
            seg = self.find_segment(t)
            init_x = self._x[seg]
            init_vel = self.knot_vels[seg]
            init_t = self._t[seg]
            delta = t - init_t
            return init_x + init_vel * delta + .5 * self._params[seg] * delta * delta
        elif order == 1:
            seg = self.find_segment(t)
            init_vel = self.knot_vels[seg]
            init_t = self._t[seg]
            delta = t - init_t
            return init_vel + self._params[seg] * delta
        elif order == 2:
            seg = self.find_segment(t)
            return self._params[seg]

    def find_segment(self, t):
        """Returns segment that the time is in, rounded down."""
        # Binary search
        # arr = [0] + list(self._t)
        # start = 0
        # end = self.num_knots
        # while (start + 1) != end:
        #     start_time = arr[start]
        #     end_time = arr[end]
        #     middle = (start_time + end_time) // 2
        #     if t < middle:
        #         end = (start + end + 1) // 2
        #     else:
        #         start = (start + end) // 2
        # return start
        for i, knot_time in enumerate(self._t):
            if t < knot_time:
                return i
        return self.num_knots - 1


if __name__ == '__main__':
    num_knots = 10
    times = np.arange(1, num_knots + 1)
    data = times * np.sin(times)
    s = ConstAccelSpline(num_knots)
    s.build_spline(times, data)
    print(s.init_vel)
    print(s._t, s._x)
    print(s.knot_vels, s.params)
    print(s.find_segment(1.3))
    ax = s.plot()
    s.plot(ax, order=1)
    s.plot(ax, order=2, show=True)

    # num_knots = 5
    # times = np.arange(1, num_knots + 1)
    # print(times)
    # s = ConstAccelSpline(num_knots)
    # s.random_spline(times, num_knots)
    # print(s._t, s._x)
    # print(s.knot_vels, s.params)
    # print(s.deriv(2.0, 1))
    # ax = s.plot()
    # s.plot(ax, order=1)
    # s.plot(ax, order=2, show=True)
