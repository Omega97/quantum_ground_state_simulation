import numpy as np
from itertools import count
from potentials.potential import Potential


class InfiniteBox(Potential):

    def __init__(self, n_states, m=1., l=np.pi/2**.5):
        self.m = m
        self.l = l
        super().__init__(n_states)

    def quantum_numbers_generator(self):
        for i in count(1, step=1):
            yield {'n': i}

    def compute_energy(self, quantum_numbers: dict):
        k = quantum_numbers['n'] * np.pi / self.l
        return np.round(k**2/2, 14)

    def compute_eigenstate(self, quantum_numbers: dict):
        a = (self.l / 2) ** .5
        k = quantum_numbers['n'] * np.pi / self.l

        def f(x):
            if x < 0 or x > self.l:
                return 0.
            else:
                return a * np.sin(k * x)

        return f


def test_1(n_states=4):
    system = InfiniteBox(n_states)
    qn = system.get_quantum_numbers()
    e = system.get_eigenvalues()
    s = system.get_eigenstates()
    print(e)
    print(s)
    print(qn)


if __name__ == '__main__':
    test_1()
