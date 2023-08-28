"""

Goal:

- Finding the eigenstates and energy spectrum of a system of identical particles


Inputs:

- 1-particle energy eigenstates
- 1-particle energy eigenvalues
- 2-particle interaction terms V_ij = V_kl = <ij|V|kl>
- type of particles (bosons, fermions, distinguishable)

"""
import numpy as np
from potentials.infinite_box import InfiniteBox
from mathematics.generators import generate_bosons, generate_fermions
from misc import sort_by_energy


class Pipeline:

    def __init__(self, n_particles, n_states, particle_type: str):
        self.n_particles = n_particles
        self.n_states = n_states
        self.particle_type = particle_type
        self.system = None
        self.quantum_numbers_1 = None
        self.energies_1 = None
        self.eigenstates_1 = None
        self.occupation_numbers = None
        self.n_config = None
        self.H_0 = None
        self.V_int = None
        self.H = None
        self.final_energies = None
        self.final_eigenstates_coefficients = None

    def compute_single_particle_states(self):
        """Compute list of quantum numbers, energy eigenvalues and eigenstates for the one-particle system"""
        self.system = InfiniteBox(self.n_states)
        self.quantum_numbers_1 = self.system.get_quantum_numbers()
        self.energies_1 = self.system.get_eigenvalues()
        self.eigenstates_1 = self.system.get_eigenstates()

    def compute_configurations(self):
        """Compute all the possible configurations of the system (depending on the type of particles)"""
        if self.particle_type.lower().startswith('boson'):
            gen = generate_bosons
        elif self.particle_type.lower().startswith('fermion'):
            gen = generate_fermions
        else:
            raise ValueError(f'Particle type {self.particle_type} unknown')
        self.occupation_numbers = tuple(gen(self.n_particles, self.n_states))
        self.n_config = len(self.occupation_numbers)  # number of configurations

    def compute_unperturbed_hamiltonian(self):
        """Use the single-particle energies to compute the unperturbed hamiltonian H_0"""
        self.H_0 = np.zeros((self.n_config, self.n_config))
        for i in range(self.n_config):
            for j in range(self.n_states):
                self.H_0[i, i] += self.occupation_numbers[i][j] * self.energies_1[j]

    def compute_configuration_interaction_energy_matrix(self, interaction_strength):
        """Compute the configuration interaction energy matrix V_(int)"""
        self.V_int = np.zeros((self.n_config, self.n_config))

        method = ...

        for i in range(self.n_config):
            for j in range(self.n_config):
                for k in ...:
                    indices = ...
                    self.V_int[i, j] += method(indices)

        self.V_int *= interaction_strength

    def compute_system_hamiltonian(self):
        """The hamiltonian of the system is H = H_0 + V_(int)"""
        self.H = self.H_0 + self.V_int

    def compute_final_eig(self):
        """Compute eigenstates and eigenvalues of H, sort them by energy"""
        self.final_energies, self.final_eigenstates_coefficients = sort_by_energy(*np.linalg.eig(self.H))

    def plot_results(self):
        """Plot the spatial distribution of the states"""
        c = self.final_eigenstates_coefficients[0]
        self.system.plot_configuration(c, self.occupation_numbers, xlim=(0, np.pi / 2 ** .5),
                                       title='System ground state')

    def execute(self):
        self.compute_single_particle_states()
        self.compute_configurations()
        self.compute_unperturbed_hamiltonian()
        self.compute_configuration_interaction_energy_matrix(interaction_strength=2.)
        print(self.V_int)

        # self.compute_system_hamiltonian()
        # self.compute_final_eig()
        # self.plot_results()


def test():
    pipeline = Pipeline(n_particles=2, n_states=2, particle_type='bosons')
    pipeline.execute()


if __name__ == '__main__':
    test()
