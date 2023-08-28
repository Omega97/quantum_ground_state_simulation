from mathematics.legendre import AssociatedLegendre
import matplotlib.pyplot as plt
import numpy as np


def test(l=11, m=3, n=101):

    f = AssociatedLegendre(l=l, m=m)

    x_ = np.linspace(-1, 1, n)
    y_ = [f(x) for x in x_]

    plt.plot(x_, y_)
    plt.title(f'AssociatedLegendre(l={l}, m={m})')
    plt.show()


test()
