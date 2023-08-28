from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from misc import normalize


class Potential:
    """ subclass this to implement quantum system """

    def __init__(self, n_states):
        self.n_states = n_states
        self.length = None
        self.quantum_numbers = None
        self.eigenvalues = None
        self.eigenstates = None
        self.compute()

    def __len__(self):
        return self.length

    def quantum_numbers_generator(self) -> dict:
        """generate dict of quantum numbers for each state"""
        raise NotImplementedError

    def compute_energy(self, quantum_numbers: dict):
        """return energy eigenvalue corresponding to those quantum_numbers"""
        raise NotImplementedError

    def compute_eigenstate(self, quantum_numbers: dict):
        """return energy eigenstate corresponding to those quantum_numbers"""
        raise NotImplementedError

    def compute(self):
        self.quantum_numbers = list(islice(self.quantum_numbers_generator(), self.n_states))
        v = [(self.compute_energy(i), self.compute_eigenstate(i)) for i in self.quantum_numbers]
        self.eigenvalues, self.eigenstates = list(zip(*v))
        self.length = len(self.quantum_numbers)

    def get_quantum_numbers(self):
        return self.quantum_numbers

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenstates(self):
        return self.eigenstates

    def linear_combination(self, state_coef):
        if len(state_coef) != len(self):
            raise ValueError(f'coefficients must be of length {len(self)} (not {len(state_coef)})')
        v = normalize(state_coef)

        def fun(x):
            return sum(v[i] * self.eigenstates[i](x) for i in range(len(self)))
        return fun

    def plot_state(self, state_coef, xlim=6, n_pt=300):
        f = self.linear_combination(state_coef)

        def f_re(x):
            return np.real(f(x))

        def f_im(x):
            return np.imag(f(x))

        fig, axs = plt.subplots(nrows=2)

        plt.sca(axs[0])

        if type(xlim) == list:
            x_min, x_max = xlim
        else:
            x_min = -xlim
            x_max = +xlim

        x_ = np.linspace(x_min, x_max, n_pt)
        y_re = f_re(x_)
        y_im = f_im(x_)
        plt.plot(x_, y_re)
        plt.plot(x_, y_im, linestyle='dashed')

        def distribution(x):
            return np.abs(f(x)) ** 2

        plt.sca(axs[1])
        y_ = distribution(x_)
        plt.plot(x_, y_)
        plt.show()

    def plot_configuration(self, config_coef, occupatin_numbers, xlim=2, n_pt=300, title='Configuration'):

        cc = normalize(config_coef)

        def distribution(x):
            out = 0.
            for i in range(len(occupatin_numbers)):
                for j in range(len(occupatin_numbers[i])):
                    p = np.abs(cc[i]) ** 2
                    y = np.abs(self.eigenstates[j](x)) ** 2
                    out += occupatin_numbers[i][j] * p * y
            return out

        if hasattr(xlim, '__len__'):
            x_min, x_max = xlim
        else:
            x_min = -xlim
            x_max = +xlim

        x_ = np.linspace(x_min, x_max, n_pt)
        y_ = [distribution(x) for x in x_]
        plt.plot(x_, y_)
        plt.title(title)
        plt.show()
