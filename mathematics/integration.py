import numpy as np
from misc import fact, get_random_point_on_sphere


class Integration:
    """
    1-D integration of a given order
    call this object with function fun and extremes x0, x1 to evaluate integral
    """
    def __init__(self, order, repeat):
        assert order >= 1 and repeat >= 1
        self.order = order
        self.repeat = repeat
        self.n = 1 + self.order * self.repeat
        self.m = None
        self.c = None
        self.w = None
        self.init_m()
        self.init_c()
        self.init_w()

    def init_m(self):
        self.m = np.zeros((self.order+1, self.order+1))
        self.m[0, 0] = 1
        for i in range(1, self.order+1):
            for j in range(self.order+1):
                self.m[i, j] = i ** j

    def init_c(self):
        v = np.array([self.order ** (i + 1) / (i + 1) for i in range(self.order+1)])
        m_inv = np.linalg.inv(self.m)
        self.c = v.dot(m_inv)

    def init_w(self):
        self.w = np.ones((self.n,), dtype=float)
        for i in range(self.n):
            self.w[i] *= self.c[i % self.order]
            if i % self.order == 0 and not (i == 0 or i == self.n - 1):
                self.w[i] *= 2

    def __call__(self, fun, x0, x1):
        dx = (x1 - x0) / (self.n - 1)
        z = np.array([fun(x0 + dx * i) for i in range(self.n)])
        return np.sum(z * self.w) * dx


class SphericalCooIntegration:
    """
    Integrate over a full sphere of given radius
    Handles singularity at the origin
    """
    def __init__(self, n_dim, order, repeat, n_sampling):
        self.n_dim = n_dim
        self.n_sampling = n_sampling
        self.alg = Integration(order, repeat)
        self.c = None
        self.set_c()

    def set_c(self):
        if self.n_dim <= 0:
            raise ValueError('self.n_dim <= 0')
        self.c = 2 * np.pi ** (self.n_dim / 2) / fact(self.n_dim / 2 - 1)

    def __call__(self, fun, radius, epsilon=10**-10):

        def g(x):
            out = 0.
            for i in range(self.n_sampling):
                v = get_random_point_on_sphere(self.n_dim)
                out += fun(*v * x)
            return out / self.n_sampling * x ** (self.n_dim - 1)

        def f(x):
            """make tha average of n points at distant x"""
            if x == 0.:
                return 2 * g(epsilon) - g(2 * epsilon)      # approximates origin
            else:
                return g(x)

        return self.alg(f, x0=0., x1=radius) * self.c
