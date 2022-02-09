from common.main.data_classes.simulation_data_classes import SimulationState
from sim.src.main.physics.sph_strategies.abstract_sph_strategy import AbstractSPHStrategy
import sim.src.main.physics.kernels as kernels
import sim.src.main.physics.constants as constants
from numba import cuda
import numpy as np
import math


class NaiveSPHStrategy(AbstractSPHStrategy):
    @property
    def next_state(self) -> SimulationState:
        return SimulationState(None, None, None, None)

    def _initialize_next_state(self, old_state):
        self.d_position = cuda.to_device(old_state.position)
        self.d_velocity = cuda.to_device(old_state.velocity)
        self.d_external_force = cuda.to_device(super().params.external_force)

        # b) arrays to be used during computations
        self.d_new_pressure_term = cuda.to_device(
            np.zeros((super().params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_viscosity_term = cuda.to_device(
            np.zeros((super().params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_density = cuda.to_device(
            np.zeros(super().params.n_particles, dtype=np.float64)
        )
        self.threads_per_grid: int = 64
        self.grids_per_block: int = math.ceil(super().params.n_particles / self.threads_per_grid)

    def _compute_density(self):
        kernels.density_kernel[self.grids_per_block, self.threads_per_grid](
            self.d_new_density,
            self.d_position,
            constants.MASS,
            constants.INF_R
        )
        cuda.synchronize()

    def _compute_pressure(self):
        kernels.pressure_kernel[self.grids_per_block, self.threads_per_grid](
            self.d_new_pressure_term,
            self.d_new_density,
            self.d_position,
            constants.MASS,
            constants.INF_R,
            constants.K,
            constants.RHO_0,
        )
        cuda.synchronize()

    def _compute_viscosity(self):
        kernels.viscosity_kernel[self.grids_per_block, self.threads_per_grid](
            self.d_new_viscosity_term,
            self.d_new_density,
            self.d_position,
            self.d_velocity,
            constants.MASS,
            constants.INF_R,
            constants.VISC,
        )
        cuda.synchronize()

    def _integrate(self):
        kernels.integrating_kernel[self.grids_per_block, self.threads_per_grid](
            self.d_position,
            self.d_velocity,
            self.d_external_force,
            self.d_new_pressure_term,
            self.d_new_viscosity_term,
            super().dt,
            constants.MASS,
        )
        cuda.synchronize()
