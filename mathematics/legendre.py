from mathematics.polynomial import Polynomial
from math import factorial as fact
import numpy as np
import matplotlib.pyplot as plt


class Legendre(Polynomial):

    """ P_l = (2**l * l!)**-1 (d/dx)**l (x**2-1)**l """

    def __init__(self, l):
        self.l = l
        p = Polynomial([-1, 0, 1]) ** l
        p = p.der(l)
        p /= (2**l * fact(l))
        super().__init__(p.c)


class AssociatedLegendre:

    """ P_(l,m) = (-1)**m (1 - x**2) ** (m/2) (d/dx)**m * P_l(x) """

    def __init__(self, l, m):
        self.l = l
        self.m = m
        self.p = Legendre(l).der(m)

    def __call__(self, x):
        m = self.m
        return (-1)**m * (1 - x**2) ** (m/2) * self.p(x)


def test_1(l=4, m=2, n_pt=201):
    p = AssociatedLegendre(l, m)
    x_ = np.linspace(0, np.pi * 2, n_pt)
    y_ = [p(np.cos(x)) for x in x_]
    y_ = np.abs(y_)

    fig, axs = plt.subplots()
    axs.plot(np.cos(x_) * y_, np.sin(x_) * y_)
    axs.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    test_1()
