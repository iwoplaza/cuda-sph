from abc import ABC, abstractmethod
import numpy as np
from numba import cuda
from common.data_classes import SimulationState, SimulationParameters
import numba.cuda.random as random
from sim.src.sph.kernels import base_kernels
from sim.src.sph.kernels.base_kernels import collision_kernel, collision_kernel_box
import sim.src.sph.thread_layout as thread_layout
from sim.src.sph.logging_utils import sph_stage_logger
import logging

logger = logging.getLogger(__name__)


class AbstractSPHStrategy(ABC):
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.dt = 1.0 / self.params.fps
        self.old_state: SimulationState = None
        self.new_state: SimulationState = None
        thread_setup = thread_layout.organize(params.n_particles)
        self.grid_size = thread_setup[0]
        self.block_size = thread_setup[1]
        self.result_force = np.zeros((self.params.n_particles, 3)).astype(np.float64)
        self.__init_log()
        self.rng_states = random.create_xoroshiro128p_states(self.grid_size * self.block_size, seed=16435234)

    @sph_stage_logger(logger=logger)
    def compute_next_state(self, old_state: SimulationState) -> SimulationState:
        self.old_state = old_state

        self._initialize_computation()
        self._compute_density()
        self._compute_pressure()
        self._compute_viscosity()
        self.__integrate()
        self.__collide()
        self.__finalize_computation()

        self.__next_state_log()
        return self.new_state

    @abstractmethod
    def _initialize_computation(self):
        pass

    @abstractmethod
    def _compute_density(self):
        pass

    @abstractmethod
    def _compute_pressure(self):
        pass

    @abstractmethod
    def _compute_viscosity(self):
        pass

    @sph_stage_logger(logger)
    def _send_arrays_to_gpu(self):
        self.d_position = cuda.to_device(self.old_state.position)
        self.d_velocity = cuda.to_device(self.old_state.velocity)
        self.d_external_force = cuda.to_device(self.params.external_force)
        self.d_new_pressure_term = cuda.to_device(
            np.zeros((self.params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_viscosity_term = cuda.to_device(
            np.zeros((self.params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_density = cuda.to_device(np.zeros(self.params.n_particles, dtype=np.float64))
        self.d_space_size = cuda.to_device(self.params.space_size)
        self.d_result_force = cuda.to_device(self.result_force)
        self.d_pipe = cuda.to_device(self.params.pipe.to_numpy())

    @sph_stage_logger(logger)
    def __finalize_computation(self):
        self.new_state = SimulationState(
            self.d_position.copy_to_host(),
            self.d_velocity.copy_to_host(),
            self.d_new_density.copy_to_host(),
        )
        self.result_force = self.d_result_force.copy_to_host()
        self.__finalize_computation_log()

    @sph_stage_logger(logger)
    def __integrate(self):
        base_kernels.integrating_kernel[self.grid_size, self.block_size](
            self.d_result_force,
            self.d_position,
            self.d_velocity,
            self.d_external_force,
            self.d_new_density,
            self.d_new_pressure_term,
            self.d_new_viscosity_term,
            self.dt,
        )
        cuda.synchronize()

    @sph_stage_logger(logger)
    def __collide(self):
        collision_kernel[self.grid_size, self.block_size](
            self.d_position,
            self.d_velocity,
            self.d_pipe,
            self.rng_states
        )
        #collision_kernel_box[self.grid_size, self.block_size](
        #    self.d_position,
        #    self.d_velocity,
        #    self.d_space_size
        #)

        cuda.synchronize()

    def __init_log(self):
        logger.info(f"Thread layout: grid size {self.grid_size}, block size {self.block_size}, dt = {self.dt}")

    def __next_state_log(self):
        logging.debug(
            f"first particle stats:"
            + f"position {np.round(self.new_state.position[0], 1).tolist()}"
            + f"velocity: {np.round(self.new_state.velocity[0], 5).tolist()}"
        )

    def __finalize_computation_log(self):
        logger.debug(
            f"Created new simulation state with: position shape: {self.new_state.position.shape}"
            + f"velocity_shape: {self.new_state.velocity.shape}"
            + f"density shape: {self.new_state.density.shape}"
        )
