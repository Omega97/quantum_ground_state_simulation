from mathematics.integration import IntegrateFunctionProduct
from potentials.qho import QHO
from generators import generate_bosons
import numpy as np
import matplotlib.pyplot as plt


def get_key(i1, j1, i2, j2):    # todo swapping coo is a symmetry
    v = sorted([str([i1, j1]), str([i2, j2])])
    return hash(str(v))
    # return hash(str([i1, i2, j1, j2]))


class ExchangeIntegrals:   # todo simplify?

    def __init__(self, states, potential, order=4, repeat=5, points=100, r0=10**-9, precision=10**-12):
        """
        Computes < psi[i1](x) * psi[j1](y) | V(x, y) | psi[i2](x) * psi[j2](y) >

        :param states: single particle states
        :param potential: interaction potential
        """
        self.states = states
        self.potential = potential
        self.data = dict()
        self.int_alg = IntegrateFunctionProduct(n_dim=2, order=order, repeat=repeat,
                                                points=points, r0=r0, precision=precision)

    def compute(self, indices):

        for i in indices:
            if i >= len(self.states):
                raise ValueError(f'Not enough states ({i} >= {len(self.states)})')

        key = get_key(*indices)

        i1, j1, i2, j2 = indices

        def psi_1(x, y):
            return self.states[i1](x) * self.states[j1](y)

        def psi_2(x, y):
            return self.states[i2](x) * self.states[j2](y)

        value = self.int_alg(psi_1, self.potential, psi_2)
        self.data[key] = value

    def __getitem__(self, indices):

        key = get_key(*indices)

        if key not in self.data:
            self.compute(indices)
        return self.data[key]


def test_1(n_states=3, points=40, n_digit=2):

    qho = QHO(n_states=n_states)
    n_, e_, states = qho()

    def V(x, y):
        return (x - y) ** 2

    ee = ExchangeIntegrals(states, V, points=points)

    on = np.array(list(generate_bosons(n_ptc=2, n_states=n_states)))
    print(on)

    config_e = [on[i].dot(np.array(e_)) for i in range(len(on))]
    print(config_e)

    def gen():
        for v in on:
            s_ = []
            for i_ in range(len(v)):
                s_ += [i_] * v[i_]
            yield s_

    w = list(gen())

    mat = np.zeros((len(w), len(w)))

    for i in range(len(w)):
        for j in range(len(w)):
            s = w[i] + w[j]
            value = ee[s]
            mat[i, j] = mat[j, i] = value

    print(np.round(mat, n_digit))

    ticks = list(generate_bosons(n_ptc=2, n_states=n_states))
    clim = max(np.abs(mat.flatten()))   # todo efficient
    plt.imshow(mat, cmap='bwr', clim=(-clim, clim))

    plt.xticks(list(range(len(ticks))), ticks, rotation=45)
    plt.yticks(list(range(len(ticks))), ticks)
    plt.show()

    return mat


def test_2(n_states=3, points=1000):

    qho = QHO(n_states=n_states)
    states = qho()[-1]

    def V(x, y):
        return (x - y) ** 2

    ee = ExchangeIntegrals(states, V, points=points)

    print(ee[0, 1, 2, 1])
    print(ee[0, 1, 1, 2])
    print(ee[1, 0, 2, 1])
    print(ee[1, 0, 1, 2])
    print(ee[1, 2, 1, 0])


if __name__ == '__main__':
    test_1()
    # test_2()
