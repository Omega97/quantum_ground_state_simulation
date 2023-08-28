import numpy as np


def equalize(v, w):
    """ returns arrays ow equal length by padding the shortest with 0s """
    m = max(len(v), len(w))
    v1 = np.pad(v, (0, m-len(v)), 'constant')
    w1 = np.pad(w, (0, m-len(w)), 'constant')
    return v1, w1


def coefficients_product(v, w):
    """find the coefficients of the product of two polynomials"""
    c = np.zeros(len(v) + len(w) - 1)
    for i in range(len(c)):
        for j in range(i + 1):
            if j < len(v) and i - j < len(w):
                c[i] += v[j] * w[i - j]
    return c


class Polynomial:
    """polynomial described by list of coefficients
    P(x) = c[0] + c[1] x + c[2] x**2 + ... """
    max_length = 25

    def __init__(self, c, dtype=np.float):
        """
        [1, 0, 3] -> 1 + 3 * x**2
        :param c: coefficients of polynomial, corresponding to increasing powers of input variable
        """
        self.c = None
        self.dtype = dtype
        self.init_coefficients(c)

    def init_coefficients(self, c):
        if not len(c):
            c = [0]
        self.c = np.array(c, dtype=self.dtype)[:Polynomial.max_length]

    def __add__(self, other):
        v, w = equalize(self.c, other.c)
        return Polynomial(v + w, dtype=self.dtype)

    def __mul__(self, other):
        if type(other) == type(self):
            c = coefficients_product(self.c, other.c)
            return Polynomial(c, dtype=self.dtype)
        else:
            return Polynomial(self.c * other, dtype=self.dtype)

    def __truediv__(self, other):
        return self * (1/other)

    def __pow__(self, power):
        if power == 0:
            return Polynomial([1], dtype=self.dtype)
        elif power == 1:
            return self
        elif power > 1:
            return np.prod([self for _ in range(power)])
        else:
            raise ValueError('Negative powers not supported!')

    def _derivative(self):
        c = self.c[1:].copy()
        for i in range(len(c)):
            c[i] *= i+1
        return Polynomial(c, dtype=self.dtype)

    def der(self, n=1):
        if n == 0:
            return self
        elif n == 1:
            return self._derivative()
        elif n > 1:
            return self.der(n-1)._derivative()
        else:
            raise ValueError

    def __call__(self, x):
        return sum([x**i * self.c[i] for i in range(len(self.c))])

    def copy(self):
        return Polynomial(self.c.copy(), dtype=self.dtype)

    def __repr__(self):
        return f'Polynomial({self.c})'
