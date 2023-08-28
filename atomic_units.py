
class Unit:
    """
    The unit is equal to self.value times the same unit expressed in IS units
    """

    def __init__(self, name: str, value=1.):
        self.name = name
        self.value = value

    def __repr__(self):
        s = f'{self.value:.5e}' if self.value != 1. else ''
        return f'{s} {self.name}'

    def __add__(self, other):
        if type(other) == Unit:
            if self.name == other.name:
                return Unit(self.name, self.value + other.value)
            else:
                raise ValueError(f'Cannot add {self} and {other}')
        else:
            raise ValueError(f'Cannot add {self} and {type(other)}')

    def __mul__(self, other):
        if type(other) == Unit:
            v = sorted([self.name, other.name])
            new_name = " ".join(v)
            return Unit(f'{new_name}', self.value * other.value)
        else:
            return Unit(self.name, self.value * other)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        return Unit(self.name + f'^{power}', value=self.value**power)


# ATOMIC UNITS

mass = Unit('aM', 9.109383632 * 10**-31)  # kg
charge = Unit('aC', 1.60217663 * 10**-19)  # C
time = Unit('aT', 2.41888 * 10**-17)  # s
length = Unit('aL', 5.29177 * 10**-11)  # m
energy = Unit('aE', 4.35974 * 10**-18)  # J
temperature = Unit('aT', 3.157751289 * 10**5)  # K
fine_structure_constant = 1 / 137.0357904


# OTHER UNITS

electron_volt = Unit('eV', 1.602 * 10**-19)  # J


# CONSTANTS

reduced_planck_constant = energy * time  # angular momentum
coulomb_constant = energy * length * charge ** -2   # J m / C^2
light_speed = 1/fine_structure_constant * length * time**-1  # m/s
boltzmann_constant = energy * temperature**-1   # entropy


# OTHER VALUES

hydrogen_gs_energy = 0.4997 * energy
helium_gs_energy = 2.9023 * energy
