"""
generate_bosons and generate_fermions are occupation number generators
"""


def _generate_state_numbers(n_ptc, n_states, _start=0, skip_first=False):

    if n_ptc == 1:
        for i in range(_start, n_states):
            yield [i]
    else:
        for i in range(_start, n_states):
            for v in _generate_state_numbers(n_ptc - 1, n_states, _start=i + skip_first, skip_first=skip_first):
                yield [i] + v


def _occupation_number_generator(skip_first):

    def wrap(n_ptc, n_states):
        for v in _generate_state_numbers(n_ptc, n_states, skip_first=skip_first):
            out = [0 for _ in range(n_states)]
            for i in range(n_ptc):
                out[v[i]] += 1
            yield out  # , tuple(sorted(v))

    return wrap


generate_bosons = _occupation_number_generator(skip_first=False)
generate_fermions = _occupation_number_generator(skip_first=True)


def _generate_distinct(n_ptc, n_states):
    if n_ptc <= 0:
        yield []
    else:
        for j in range(n_states):
            for i in _generate_distinct(n_ptc-1, n_states):
                yield i + [j]


def generate_distinct(n_ptc, n_states):
    for v in _generate_distinct(n_ptc, n_states):
        out = [0 for _ in range(n_states)]
        for i in v:
            out[i] += 1
        yield out  # , tuple(v)


def test():

    v = (generate_distinct(2, 3),
         generate_bosons(2, 3),
         generate_fermions(2, 3))

    for g in v:
        for j in g:
            print('  '.join([str(i) for i in j]))
        print()


if __name__ == '__main__':
    test()
