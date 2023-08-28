"""Hermite polynomial coefficients"""
import itertools
from mathematics.polynomial import Polynomial
from numpy import longlong


class HermiteItem:
    """Represents a gaussian * a polynomial"""
    def __init__(self, p: Polynomial):
        self.p = p

    def der(self):
        return HermiteItem(self.p.der() + self.p * Polynomial(c=[0, -2], dtype=longlong))

    def __mul__(self, other):
        return HermiteItem(self.p * other)

    def get_coefficients(self):
        return self.p.c


def hermite_coefficient_gen():
    """generator of coefficients of (physicist's) Hermite polynomials"""
    a = HermiteItem(p=Polynomial([1], dtype=longlong))
    while True:
        yield a.get_coefficients()
        a = a.der() * -1


def hermite_polynomial_gen():
    """generator of coefficients of (physicist's) Hermite polynomials"""
    a = HermiteItem(p=Polynomial([1], dtype=longlong))
    while True:
        yield a.p
        a = a.der() * -1


def hermite_list(n):
    """list of (physicist's) Hermite polynomials"""
    return list(itertools.islice(hermite_polynomial_gen(), n))


def example():
    for i, v in enumerate(hermite_coefficient_gen()):
        print(f'[{i}]\n')
        for j in v:
            print(f'{j:20}')
        print('\n')
        if v[0] > 1000:
            break


if __name__ == '__main__':
    example()
