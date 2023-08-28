import numpy as np


def fact(x):
    """analytical extension of factorial"""
    if x == 0:
        return 1.
    elif x == -1/2:
        return np.pi ** .5
    elif x > 0:
        return fact(x-1) * x
    else:
        raise ValueError(f'fact({x}) not defined')


def get_random_point_on_sphere(n_dim):
    v = np.random.normal(size=n_dim)
    m = np.linalg.norm(v)
    if m:
        return v / m
    else:
        return get_random_point_on_sphere(n_dim)


def sort_by_energy(eigenvalues, eigenstates):
    """sort eigenvalues, perform the same operations on eigenstates"""
    v = list(zip(eigenvalues, eigenstates))
    v.sort(key=lambda w: w[0])
    return list(zip(*v))


def normalize(v):
    v = np.array(v)
    return v / v.conj().dot(v)


class Store(dict):

    def transform_key(self, key):
        """overwrite to modify the key for __getitem__ and __setitem__"""
        return key

    def compute(self, key):
        """takes the unmodified key as input, returns the value to store in self"""
        raise NotImplementedError

    def __setitem__(self, key, value):
        new_key = self.transform_key(key)
        super().__setitem__(new_key, value)

    def __getitem__(self, key):
        new_key = self.transform_key(key)
        if new_key not in self:
            super().__setitem__(new_key, self.compute(key))
        return super().__getitem__(new_key)


def newton_method(function, x0, precision, n_steps=100, dx=.01):
    x = x0
    for i in range(n_steps):
        y = function(x)

        if abs(y) <= precision:
            return x

        der = (function(x+dx) - function(x-dx)) / (2 * dx)
        if der == 0:
            x = x0 * (np.random.random() - np.random.random())
        x -= y / der

    raise ValueError(f'Algorithm did not converge after {n_steps} steps')


def dichotomic_search(function, x_min, x_max, precision, n_steps, k=.4):
    x_avg = (x_min + x_max) / 2
    for i in range(n_steps):
        y = function(x_avg)


        print()
        print(x_min, function(x_min))
        print(x_avg, function(x_avg))
        print(x_max, function(x_max))


        if abs(y) <= precision:

            print(f'dichotomic_search steps = {i}')
            print(f'f({x_avg}) = {function(x_avg)}')
            print()

            return x_avg
        elif y * function(x_min) > 0:
            x_min = x_min + (x_avg - x_min) * k
        elif y * function(x_max) > 0:
            x_max = x_max + (x_avg - x_max) * k
        else:
            raise ValueError
        x_avg = (x_min + x_max) / 2
    return x_avg


def find_min(function, precision, n_steps, x0, k=.8):
    for i in range(n_steps):
        x0 *= k
        if np.abs(function(x0)) > precision:

            print(f'find_min steps = {i}')
            print(f'f({x0}) = {function(x0)}')
            print()

            return x0
    else:
        raise ValueError(f'Function does not converge to 0 (f({x0} = {function(x0)}))')


def find_max(function, precision, n_steps, x0, k=1.5):
    for i in range(n_steps):
        x0 *= k
        if np.abs(function(x0)) <= precision:

            print(f'find_max steps = {i}')
            print(f'f({x0}) = {function(x0)}')
            print()

            return x0
    else:
        raise ValueError('Function does not converge to 0')


def find_radius(function, precision, x0=1., n_steps=20):
    x_max = find_max(function, precision, n_steps, x0=x0)
    x_min = find_min(function, precision, n_steps, x0=x_max)

    def f(x):
        return abs(function(x)) - precision

    return dichotomic_search(f, x_min, x_max, precision/2, n_steps)
