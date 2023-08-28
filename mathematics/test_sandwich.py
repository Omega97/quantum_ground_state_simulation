import numpy as np
from sandwich import RadialSandwichSolver


def test_RadialSandwichSolver(order=5, repeat=5, n_sampling=20, n_states=6):

    def potential(d):
        return 1 / d

    def f(n):
        def wrap(x):
            return np.sin(n * x * np.pi) if 0 <= x <= np.pi else 0.
        return wrap

    functions = [f(i+1) for i in range(n_states)]

    solver = RadialSandwichSolver(functions=functions, potential=potential,
                                  order=order, repeat=repeat, n_sampling=n_sampling)

    a = solver.compute(indices=(1, 2, 3, 4))
    print(a)


if __name__ == '__main__':
    test_RadialSandwichSolver()
