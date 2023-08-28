from integration import SphericalCooIntegration
from misc import get_random_point_on_sphere, find_radius


class RadialSandwichSolver:

    def __init__(self, functions, potential, order, repeat, n_sampling, precision=10**-8):
        self.functions = functions
        self.potential = potential
        self.n_sampling = n_sampling
        self.precision = precision
        self.integration_alg = SphericalCooIntegration(n_dim=2, order=order, repeat=repeat, n_sampling=n_sampling)

        self.function_2d = None
        self.radius = None

        assert n_sampling >= 1

    def _compute_function_2d(self, indices):
        assert len(indices) == 4
        for i in indices:
            assert 0 <= i < len(self.functions)
        i1, i2, j1, j2 = indices

        def function(u, y):
            out = self.functions[i1](u+y) * self.functions[i2](y)
            out = out.conjugate()
            out *= self.potential(u)
            out *= self.functions[j1](u+y) * self.functions[j2](y)
            return out

        self.function_2d = function

    def _compute_optimal_radius(self):
        v_ = [get_random_point_on_sphere(n_dim=2) for _ in range(self.n_sampling)]

        def f_max(x):
            w = [abs(self.function_2d(*v * x)) for v in v_]
            return max(w)

        self.radius = find_radius(f_max, precision=self.precision)
        print(f'r = {self.radius}')

    def compute(self, indices):
        self._compute_function_2d(indices)
        self._compute_optimal_radius()
        return self.integration_alg(self.function_2d, self.radius)
