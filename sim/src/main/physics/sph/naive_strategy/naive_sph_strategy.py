from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters
from sim.src.main.physics.sph.base_strategy.abstract_sph_strategy import AbstractSPHStrategy
import sim.src.main.physics.constants as constants
from numba import cuda
from sim.src.main.physics.sph.naive_strategy import kernels


class NaiveSPHStrategy(AbstractSPHStrategy):

    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def _initialize_computation(self):
        super()._send_arrays_to_gpu()

    def _compute_density(self):
        kernels.density_kernel[self.grid_size, self.block_size](
            self.d_new_density,
            self.d_position,
            constants.MASS,
            constants.INF_R
        )
        cuda.synchronize()

    def _compute_pressure(self):
        kernels.pressure_kernel[self.grid_size, self.block_size](
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
        kernels.viscosity_kernel[self.grid_size, self.block_size](
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
        kernels.integrating_kernel[self.grid_size, self.block_size](
            self.d_position,
            self.d_velocity,
            self.d_external_force,
            self.d_new_pressure_term,
            self.d_new_viscosity_term,
            self.dt,
            constants.MASS,
        )
        cuda.synchronize()

    def _finalize_computation(self):
        self.new_state = SimulationState(
            self.d_position.copy_to_host(),
            self.d_velocity.copy_to_host(),
            self.d_new_density.copy_to_host(),
        )
