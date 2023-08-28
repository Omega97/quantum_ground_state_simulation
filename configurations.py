import numpy as np
from generators import *
from potentials.qho import QHO
import matplotlib.pyplot as plt
from exchange_integrals import ExchangeIntegrals


class Configurations:

    """
    Given a system of interacting particles:
    - Computes configuration energy (of the non-interactive version of the system)
    - Computes the interaction potential of the same system
    """

    def __init__(self, occupation_numbers, external_potential, interaction_potential, order=4, repeat=5, points=100):
        """
        The hamiltonian operator will be expressed in the base of all the possible configuration of the system

        :param occupation_numbers:
        :param external_potential:
        :param interaction_potential: V_int(x, y)
        """
        self.occupation_numbers = list(occupation_numbers)
        self.external_potential = external_potential
        self.n_states = external_potential.n_states
        self.quantum_numbers, self.eigenvalues, self.eigenstates = self.external_potential()   # for single particle
        self.interaction = interaction_potential
        self.exchange_energy = ExchangeIntegrals(self.eigenstates, interaction_potential,
                                                 order=order, repeat=repeat, points=points)
        self.H_0 = None
        self.V = None
        self.config_energies = None

    def compute_config_v_mat(self):
        n_config = len(self.occupation_numbers)
        self.V = np.zeros((n_config, n_config))
        for i in range(n_config):
            for j in range(i+1):
                for i1 in range(self.n_states):
                    for j1 in range(i1):
                        n1 = self.occupation_numbers[i][i1]
                        n2 = self.occupation_numbers[j][j1]
                        a = n1 * n2 * self.V[i1, j1]
                        self.V[i, j] += a

                    n0 = self.occupation_numbers[i][i1]
                    a = n0 * (n0-1) // 2 * self.V[i1, i1]
                    self.V[i, j] += a

                self.V[j, i] = self.V[i, j]

    def compute_config_e_mat(self):
        n_config = len(self.occupation_numbers)
        self.config_energies = np.zeros(n_config)

        for i in range(n_config):
            for i1 in range(self.n_states):
                n = self.occupation_numbers[i][i1]
                self.config_energies[i] += self.eigenvalues[i1] * n

        self.H_0 = np.identity(n_config) * self.config_energies

    def sort(self):
        """sort energies and rearrange rows and columns of H_0 and V accordingly"""
        self.config_energies, w = sorted_and_new_indices(self.config_energies)
        self.occupation_numbers.sort()
        self.H_0 = self.H_0[:, w][w]
        self.V = self.V[:, w][w]

    def __call__(self):
        self.compute_config_e_mat()
        self.compute_config_v_mat()
        self.sort()
        return self.config_energies, self.H_0, self.V


def sorted_and_new_indices(v):
    w = [(i, v[i]) for i in range(len(v))]
    w.sort(key=lambda x: x[-1])
    v = [i[1] for i in w]
    w = [i[0] for i in w]
    return v, w


def test_1(n_ptc=2, n_states=3, order=4, repeat=25, points=20, k_int=(0, .1, -.1)):

    occupation_numbers = list(generate_bosons(n_ptc, n_states))

    potential = QHO(n_states=n_states)

    def interaction_potential(x, y):
        return (x - y) ** 2

    cfg = Configurations(occupation_numbers, potential, interaction_potential,
                         order=order, repeat=repeat, points=points)
    energies, H_0, V = cfg()

    energies, w = sorted_and_new_indices(energies)

    occupation_numbers.sort()
    H_0 = H_0[:, w][w]
    V = V[:, w][w]

    for k in k_int:
        H = H_0 + k * V

        sol_e, sol_v = np.linalg.eig(H)
        sol_v = sol_v / sum(sol_v ** 2) ** .5  # normalization

        sol_v = sol_v.T

        sol_e, w = sorted_and_new_indices(sol_e)
        sol_v = sol_v[w]  # todo check

        plt.title(f'Interacting particles in harmonic oscillator\n'
                  f'n_ptc={n_ptc}, n_states={n_states}, k_int={k}')
        plt.imshow(sol_v**2, cmap='bwr', clim=(-1, 1))
        plt.gca().invert_yaxis()
        r = list(range(len(sol_e)))
        plt.xlabel('Configurations')
        plt.ylabel('Energy')
        x_ticks = [' '.join([str(j) for j in i]) for i in occupation_numbers]
        plt.xticks(r, x_ticks, rotation=45)
        plt.yticks(r, np.round(sol_e, 3))
        plt.show()


def test_2(x_max=1., n_pt=51):

    H0 = np.array([[1, 0, 0],
                   [0, 2, 0],
                   [0, 0, 3]])

    V = np.array([[1, 0, -1],
                  [0, 1, 0],
                  [-1, 0, 3]])

    data_e = [[] for _ in V]
    data_gs = [[] for _ in V]

    k_ = np.linspace(-x_max, x_max, n_pt)

    for k in k_:
        H = H0 + V * k
        E_, psi_ = np.linalg.eig(H)

        psi_ = psi_.T

        E_, w = sorted_and_new_indices(E_)
        psi_ = psi_[w]
        psi_ = psi_ ** 2
        psi_ = psi_ / sum(psi_)

        for i in range(len(data_e)):
            data_e[i] += [E_[i]]
            data_gs[i] += [psi_[0][i]]

    fig, axs = plt.subplots(nrows=2, ncols=1)

    plt.sca(axs[0])
    for v in data_e:
        plt.plot(k_, v)

    plt.sca(axs[1])
    plt.imshow(data_gs, cmap='bwr', clim=(-1, +1))
    plt.xticks(list(range(n_pt)), np.round(k_, 3), rotation=45)
    plt.show()


def test_3(n_ptc=2, n_states=2, order=5, repeat=10, points=100):

    occupation_numbers = list(generate_bosons(n_ptc, n_states))

    potential = QHO(n_states=n_states)

    def interaction_potential(x, y):
        return (x - y) ** 2

    cfg = Configurations(occupation_numbers, potential, interaction_potential,
                         order=order, repeat=repeat, points=points)
    energies, H_0, V = cfg()

    print(np.round(H_0, 3), '\n')
    print(np.round(V, 3), '\n')


if __name__ == '__main__':
    # test_1(k_int=np.array([.0, .001, .01, .03, .1, .3, 1., 3]))
    # test_2()
    test_3()
