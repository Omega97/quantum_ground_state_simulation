import numpy as np
import matplotlib.pyplot as plt
from potentials.qho import QHO


def sort_with_indices(elements: list, **kwargs):
    """ returns sorted list of elements and list of old indices """
    a = zip(elements, range(len(elements)))
    a = sorted(a, **kwargs)
    v1, v2 = zip(*a)
    return v1, v2


class InteractingParticleStates:
    """
    Takes the configuration energies and interaction potential matrix as input
    When called it returns the eigenvalues and eigenstates of energy in the space of configurations
    """

    def __init__(self, config_energies, interaction_potential_matrix):
        self.n_config = len(config_energies)
        self.config_energies = np.array(config_energies)
        self.V_int = interaction_potential_matrix
        self.H_0 = np.identity(self.n_config) * self.config_energies
        self.H = None
        self.sort()

    def sort(self):
        """sort energies and rearrange rows and columns of H_0 and V accordingly"""
        self.config_energies, w = sort_with_indices(self.config_energies)
        self.H_0 = self.H_0[:, w][w]
        self.V_int = self.V_int[:, w][w]

    def __call__(self, k=1.):
        """compute eigenvalues and eigenstates of energy
        eigenvalues are returned in ascending order (and eigenstates are re-arranged accordingly)
        """
        self.H = self.H_0 + self.V_int * k
        config_eigenvalues, config_eigenstates = np.linalg.eig(self.H)
        config_eigenstates = config_eigenstates.T
        config_eigenvalues, w = sort_with_indices(config_eigenvalues)
        config_eigenstates = config_eigenstates[w]
        return config_eigenvalues, config_eigenstates


def test_1(k=1., n_points=401, x_max=4.):
    # config_energies = [1, 2, 3]
    #
    # V_int = np.array([[1, 0, -1],
    #                   [0, 1, 0],
    #                   [-1, 0, 3]])
    #
    # state_indices = [[0, 0], [0, 1], [1, 1]]

    config_energies = [1, 2, 3, 3, 4, 5]
    n_config = len(config_energies)

    V_int = np.array([[1., 0., 0.7, -1., 0., 0.],
                      [0., 2., 0., 0., -1.3, -.1],
                      [0.7, 0., 3., -1.4, 0., .7],
                      [-1., 0., -1.4, 3., 0., -2.1],
                      [0., -1.3, 0., 0., 4., 0.],
                      [0., -.1, .7, -2.1, 0., 5.]])

    state_indices = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    ips = InteractingParticleStates(config_energies=config_energies, interaction_potential_matrix=V_int)
    int_energies, int_states = ips(k)

    print('\nint energies')
    print(int_energies)
    print('\nint states')
    for v in int_states:
        print(np.round(v, 3))

    x_ = np.linspace(-x_max, x_max, n_points)

    qho = QHO(n_states=3)
    psi0 = qho()[-1]

    fig, axs = plt.subplots(ncols=n_config)

    for n in range(n_config):
        def f(x):
            return sum((int_states[n][i] ** 2 * (psi0[state_indices[i][0]](x) ** 2 + psi0[state_indices[i][1]](x) ** 2))
                       for i in range(n_config))

        y_ = [f(x) for x in x_]
        axs[n].plot(x_, y_)
        axs[n].fill_between(x_, y_, alpha=.5)
        axs[n].set_aspect(10)
        axs[n].set_title(f'#{n} excited state' if n else 'Ground state')

    # if k == 0:
    #     s = 'non-interacting'
    # else:
    #     s = ', harmonic ' + ('attraction' if k > 0 else 'repulsion')
    # plt.title(f'Harmonic oscillator{s}, k={k:.1f}')
    plt.show()


def test_2(k_max=2., n_pt=201):
    energies = [1, 2, 3]

    V = np.array([[1, 0, -1],
                  [0, 1, 0],
                  [-1, 0, 3]])

    ips = InteractingParticleStates(config_energies=energies, interaction_potential_matrix=V)

    data_e = [[] for _ in V]
    data_gs = [[] for _ in V]

    k_ = np.linspace(-k_max, k_max, n_pt)

    for k in k_:

        config_energies, states = ips(k)

        for i in range(len(data_e)):
            data_e[i] += [config_energies[i]]
            data_gs[i] += [states[0][i] ** 2]

    fig, axs = plt.subplots(nrows=2, ncols=1)

    plt.sca(axs[0])
    for v in data_e:
        plt.plot(k_, v)

    plt.sca(axs[1])
    plt.imshow(data_gs, cmap='bwr', clim=(-1, +1), aspect=10, interpolation='None')

    n_xticks = 9
    xticks = np.round(np.linspace(-k_max, k_max, n_xticks), 2)
    plt.xticks(np.linspace(0, n_pt - 1, n_xticks), xticks, rotation=45)
    n_yticks = len(energies)
    yticks = list(range(n_yticks))
    plt.yticks(np.linspace(0, n_yticks - 1, n_yticks), yticks)

    plt.show()


if __name__ == '__main__':
    test_1(-1.)

    # test_2()
