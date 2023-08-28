from math import factorial
import numpy as np
from itertools import count
from mathematics.hermite import hermite_list
from potentials.potential import Potential


class QHO(Potential):

    def __init__(self, n_states, m=1., frequency=1.):
        """ Quantum Harmonic Oscillator """
        self.m = m
        self.frequency = frequency
        self.hermite_p = hermite_list(n_states)
        super().__init__(n_states)

    def quantum_numbers_generator(self):
        for i in count(0, step=1):
            yield {'n': i}

    def compute_energy(self, quantum_numbers: dict):
        n = quantum_numbers['n']
        return self.frequency * (n+1/2)

    def compute_eigenstate(self, quantum_numbers: dict):
        n = quantum_numbers['n']
        k = self.m * self.frequency
        c = (2 ** n * factorial(n)) ** (-1 / 2) * (k / np.pi) ** (1 / 4)

        def f(p):
            def wrap(x):
                return c * np.exp(-k / 2 * x**2) * p(k**.5 * x)
            wrap.__name__ = f'qho_{n}'
            return wrap

        return f(p=self.hermite_p[n])


def test_1(n_states=3):
    qho = QHO(n_states)
    qn = qho.get_quantum_numbers()
    e = qho.get_eigenvalues()
    s = qho.get_eigenstates()
    print(e)
    print(s)
    print(qn)


def test_2(xlim=7, n_pt=300, n_states=4):
    system = QHO(n_states)
    system.plot([1, complex(0, 1), 0, 0], xlim, n_pt)


if __name__ == '__main__':
    test_2()
