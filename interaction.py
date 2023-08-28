from misc import Store
import numpy as np

np.random.seed(0)


class StoreTerms(Store):

    def transform_key(self, key: tuple):
        if not hasattr(key, '__len__'):
            raise ValueError(f'{key} has no len')
        assert len(key) == 4

        def symmetry_1(a, b, c, d):
            return b, a, d, c

        def symmetry_2(a, b, c, d):
            return c, d, a, b

        symmetries = [symmetry_1, symmetry_2]   # list of symmetries to respect

        v = [key]                               # list of all symmetries od the object
        for s in symmetries:
            for i in range(len(v)):
                v += [s(*v[i])]

        v = [str(hash(i)) for i in v]
        v.sort()                                # now every equivalent object yields the same v
        return hash(' '.join(v))

    def compute(self, key):
        print(f'computing {key}')
        return complex(np.random.random(), np.random.random())

    def __getitem__(self, key):
        assert len(key) == 4
        if key[0] <= key[2]:
            return super().__getitem__(key)
        else:
            return super().__getitem__(key).conjugate()


def test_key():
    store = StoreTerms()

    keys = [(1, 2, 3, 4),
            (2, 1, 4, 3),
            (3, 4, 1, 2),
            (4, 3, 2, 1),
            (2, 1, 0, 0)]

    for key in keys:
        print(store.transform_key(key))


def test_compute():
    store = StoreTerms()

    keys = [(1, 2, 3, 4),
            (2, 1, 4, 3),
            (3, 4, 1, 2),
            (4, 3, 2, 1)]

    for key in keys:
        print(key, store[key])


if __name__ == '__main__':
    test_compute()
