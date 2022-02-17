from common.data_classes import SimulationParameters
from sim.src.sph.strategies.abstract_sph_strategy import AbstractSPHStrategy
from numba import cuda
from sim.src.sph.kernels import naive_kernels
from sim.src.sph.logging_utils import sph_stage_logger
import logging 

logger = logging.getLogger(__name__)

class NaiveSPHStrategy(AbstractSPHStrategy):

    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    @sph_stage_logger(logger=logger)
    def _initialize_computation(self):
        super()._send_arrays_to_gpu()

    @sph_stage_logger(logger=logger)
    def _compute_density(self):
        naive_kernels.density_kernel[self.grid_size, self.block_size](
            self.d_new_density,
            self.d_position,
        )
        cuda.synchronize()

    @sph_stage_logger(logger=logger)
    def _compute_pressure(self):
        naive_kernels.pressure_kernel[self.grid_size, self.block_size](
            self.d_new_pressure_term,
            self.d_new_density,
            self.d_position,
        )
        cuda.synchronize()

    @sph_stage_logger(logger=logger)
    def _compute_viscosity(self):
        naive_kernels.viscosity_kernel[self.grid_size, self.block_size](
            self.d_new_viscosity_term,
            self.d_new_density,
            self.d_position,
            self.d_velocity,
        )
        cuda.synchronize()


