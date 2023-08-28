from time import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from mathematics.integration import IntegrateFunctionProduct
from potentials.qho import QHO
from generators import generate_bosons


def test_1():

    qho = QHO(1)

    def f(x, y, k=4):
        # return (x**2 + y**2) ** .5
        return np.exp((x - y) ** 2 * k)

    t = time()
    _, psi = qho[0]
    res = IntegrateFunctionProduct(n_dim=2, order=5, repeat=5, points=20)(psi, f, psi)

    c = (1 / pi) ** (1 / 4)
    sol = c ** 2 * (2 * pi**3)**.5

    print(res)
    print(abs(res-sol))

    print(time()-t)


def test_2(n_states=3, order=4, repeat=20, points=20):

    qho = QHO(n_states)
    e0, psi0 = qho[0]

    def V(x, y):
        return (x - y) ** 2

    sandwich = IntegrateFunctionProduct(n_dim=2, order=order, repeat=repeat, points=points)

    def state(x, y):
        return psi0(x) * psi0(y)

    out = sandwich(state, V, state)
    print(out)


def test_3(n_states=5, order=4, repeat=20, points=20):

    qho = QHO(n_states)

    v_mat = np.zeros((n_states, n_states))
    e_list = np.zeros(n_states)

    def V(x, y):
        return (x - y)**2

    sandwich = IntegrateFunctionProduct(n_dim=2, order=order, repeat=repeat, points=points)

    for i in range(n_states):
        e_list[i], psi1 = qho[i]
        for j in range(i+1):
            _, psi2 = qho[j]

            def f(x, y):
                return psi1(x) * psi2(y)

            v_mat[i, j] = v_mat[j, i] = sandwich(f, V, f)

    print(e_list)
    print(np.round(v_mat, 3))

    v_clim = np.max(abs(v_mat.flatten()))
    plt.imshow(v_mat, cmap='bwr', clim=[-v_clim, v_clim])
    plt.title('Exchange energies')
    plt.show()

    return e_list, v_mat


def test_4(n_states=2, order=4, repeat=20, points=20):

    def compute():  # todo check

        qho = QHO(n_states)

        v_mat = np.zeros((n_states, n_states))
        e_list = np.zeros(n_states)

        def g(x, y):
            d = (x**2 + y**2) ** .5
            return d

        sandwich = IntegrateFunctionProduct(n_dim=2, order=order, repeat=repeat, points=points)

        for i_ in range(n_states):
            e_list[i_], psi1 = qho[i_]
            for j_ in range(i_+1):
                _, psi2 = qho[j_]

                def f(x, y):
                    return psi1(x) * psi2(y)

                v_mat[i_, j_] = v_mat[j_, i_] = sandwich(f, g, f)

        return e_list, v_mat

    e_, w = compute()

    states = list(generate_bosons(2, n_states))
    m_ptc_states = len(states)

    E = np.zeros((m_ptc_states, m_ptc_states))

    print(states)

    for i in range(m_ptc_states):
        for j in range(n_states):
            E[i, i] += states[i][j] * e_[j]

    print(E)

    w_mat = np.zeros((m_ptc_states, m_ptc_states))
    for i in range(m_ptc_states):
        for j in range(i, m_ptc_states):
            print(i, j)

    print(w_mat)


def test_5(n_states=2, order=4, repeat=20, points=20):

    qho = QHO(n_states)

    def V(x, y):
        return (x-y)**2

    sandwich = IntegrateFunctionProduct(n_dim=2, order=order, repeat=repeat, points=points)

    e_, f_ = qho()

    def psi(i, j):
        def wrap(x, y):
            return f_[i](x) * f_[j](y)
        return wrap

    out = sandwich(psi(1, 1), V, psi(0, 1))
    print(out)


if __name__ == '__main__':
    test_3()
