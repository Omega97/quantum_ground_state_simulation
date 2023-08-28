import numpy as np
from integration import SphericalCooIntegration


def test_integration():
    ...


def test_SphericalCooIntegration(order, repeat, n_sampling, epsilon):

    alg = SphericalCooIntegration(order=order, repeat=repeat, n_sampling=n_sampling, n_dim=2)

    def r(*x):
        return sum(i**2 for i in x) ** .5

    def fun_1(*_):
        return 1 / np.pi

    def fun_2(x, y):
        return 1 / (2 * np.pi * r(x, y))

    def fun_3(x, y):
        return np.sin(r(x, y) * np.pi) / 2

    print('pErr:')
    for f in [fun_1, fun_2, fun_3]:
        out = alg(f, radius=1, epsilon=epsilon)
        err = abs(out - 1)
        if err == 0:
            print('perfect!')
        else:
            print(f'{-np.log10(err):6.2f}')


if __name__ == '__main__':
    test_SphericalCooIntegration(repeat=4, order=5, n_sampling=1, epsilon=10**-10)
