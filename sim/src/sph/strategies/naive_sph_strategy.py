from common.data_classes import SimulationParameters
from config import MASS, INF_R, K, RHO_0, VISC
from sim.src.sph.strategies.abstract_sph_strategy import AbstractSPHStrategy
from numba import cuda
from sim.src.sph.kernels import naive_kernels


class NaiveSPHStrategy(AbstractSPHStrategy):

    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def _initialize_computation(self):
        super()._send_arrays_to_gpu()

    def _compute_density(self):
        naive_kernels.density_kernel[self.grid_size, self.block_size](
            self.d_new_density,
            self.d_position,
            MASS,
            INF_R
        )
        cuda.synchronize()

    def _compute_pressure(self):
        naive_kernels.pressure_kernel[self.grid_size, self.block_size](
            self.d_new_pressure_term,
            self.d_new_density,
            self.d_position,
            MASS,
            INF_R,
            K,
            RHO_0,
        )
        cuda.synchronize()

    def _compute_viscosity(self):
        naive_kernels.viscosity_kernel[self.grid_size, self.block_size](
            self.d_new_viscosity_term,
            self.d_new_density,
            self.d_position,
            self.d_velocity,
            MASS,
            INF_R,
            VISC,
        )
        cuda.synchronize()


