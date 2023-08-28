from misc import *


def test_sort_by_energy():
    v = (3, 1, 2)
    w = ([3, 2], [1, 3], [2, 1])
    print(sort_by_energy(v, w))


def test_store():

    class SubStore(Store):

        def compute(self, key):
            print('computing...')
            return f'{key} value'

        def transform_key(self, key):
            return key[0]

    sub_store = SubStore()
    print(sub_store['a'])
    print(sub_store['a'])
    print(sub_store['a2'])
    print(sub_store)


def test_newton():

    def function(x):
        return np.exp(-x) - 10**-8

    x0 = newton_method(function, x0=1., precision=10**-5)
    print(x0)
    print(function(x0))


def test_find_radius():
    def f(x):
        return np.exp(-x) * np.sin(3*x)

    x0 = find_radius(f, precision=10 ** -4, x0=1.11)
    print(x0)


if __name__ == '__main__':
    # test_sort_by_energy()
    # test_store()
    test_find_radius()
