from numpy import pi, cos, exp
import numpy as np
from math import factorial as fact
from mathematics.legendre import AssociatedLegendre
import matplotlib.pyplot as plt
from mathematics.integration import get_random_point_on_sphere


class SphericalHarmonics:

    def __init__(self, l, m):
        self.l = l
        self.m = abs(m)
        self.p = AssociatedLegendre(l, self.m)

    def __call__(self, theta, phi):
        l = self.l
        m = self.m
        a = (2 * l + 1) * fact(l - m)
        a /= (2 * pi * fact(l + m))
        a = a ** .5 * exp(np.complex(0, m * phi))
        a *= self.p(cos(theta))
        return a


def test_1(l=0, m=0, n_pixel=70, n_pt=5 * 10**4):

    fig, axs = plt.subplots(ncols=3)

    sh = SphericalHarmonics(l=l, m=m)

    def f(theta_, phi_):
        return np.real(sh(theta_, phi_))

    mat_x = np.zeros((n_pixel, n_pixel))
    mat_y = np.zeros((n_pixel, n_pixel))
    mat_z = np.zeros((n_pixel, n_pixel))

    num_x = np.zeros((n_pixel, n_pixel))
    num_y = np.zeros((n_pixel, n_pixel))
    num_z = np.zeros((n_pixel, n_pixel))

    for _ in range(n_pt):
        x, y, z = get_random_point_on_sphere(3)
        theta = np.arccos(z)
        phi = np.arctan(y/x)
        i = int(round((n_pixel - 1) * (x+1)/2))
        j = int(round((n_pixel - 1) * (y+1)/2))
        k = int(round((n_pixel - 1) * (z+1)/2))
        value = f(theta, phi)

        mat_x[i, j] += value
        mat_y[j, k] += value
        mat_z[k, i] += value

        num_x[i, j] += 1
        num_y[j, k] += 1
        num_z[k, i] += 1

    mat_x = mat_x / (num_x + 1)
    mat_y = mat_y / (num_y + 1)
    mat_z = mat_z / (num_z + 1)

    clim = max([max(np.abs(mat.flatten())) for mat in (mat_x, mat_y, mat_z)]) * 2

    for mat, i in ((mat_x, 0), (mat_y, 1), (mat_z, 2)):
        # clim = max(mat.flatten())
        axs[i].imshow(mat, cmap='bwr', clim=(-clim, clim), interpolation='none')
        axs[i].axis('off')

    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    test_1(l=4, m=3)
