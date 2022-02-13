from abc import ABC, abstractmethod
import numpy as np
from numba import cuda
from common.data_classes import SimulationState, SimulationParameters
from sim.src.sph.kernels import base_kernels
from sim.src.sph.kernels.base_kernels import collision_kernel_box
import sim.src.sph.thread_layout as thread_layout


class AbstractSPHStrategy(ABC):
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.dt = 1 / self.params.n_particles
        self.old_state: SimulationState = None
        self.new_state: SimulationState = None
        thread_setup = thread_layout.organize(params.n_particles)
        self.grid_size = thread_setup[0]
        self.block_size = thread_setup[1]
        self.result_force = np.zeros((self.params.n_particles, 3)).astype(np.float64)

    def compute_next_state(self, old_state: SimulationState) -> SimulationState:
        self.old_state = old_state

        self._initialize_computation()
        self._compute_density()
        self._compute_pressure()
        self._compute_viscosity()

        self.__integrate()
        self.__collide()
        self.__finalize_computation()
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

    def _send_arrays_to_gpu(self):
        self.d_position = cuda.to_device(self.old_state.position)
        self.d_velocity = cuda.to_device(self.old_state.velocity)
        self.d_external_force = cuda.to_device(self.params.external_force)
        self.d_new_pressure_term = cuda.to_device(np.zeros((self.params.n_particles, 3), dtype=np.float64))
        self.d_new_viscosity_term = cuda.to_device(np.zeros((self.params.n_particles, 3), dtype=np.float64))
        self.d_new_density = cuda.to_device(np.zeros(self.params.n_particles, dtype=np.float64))
        self.d_space_size = cuda.to_device(self.params.space_size)
        self.d_result_force = cuda.to_device(self.result_force)

    def __finalize_computation(self):
        self.new_state = SimulationState(
            self.d_position.copy_to_host(),
            self.d_velocity.copy_to_host(),
            self.d_new_density.copy_to_host(),
        )
        self.result_force = self.d_result_force.copy_to_host()

    def __integrate(self):
        base_kernels.integrating_kernel[self.grid_size, self.block_size](
            self.d_result_force,
            self.d_position,
            self.d_velocity,
            self.d_external_force,
            self.d_new_pressure_term,
            self.d_new_viscosity_term,
            self.dt,
        )
        cuda.synchronize()

    def __collide(self):
        collision_kernel_box[self.grid_size, self.block_size](
            self.d_position,
            self.d_velocity,
            self.d_space_size
        )
        cuda.synchronize()

