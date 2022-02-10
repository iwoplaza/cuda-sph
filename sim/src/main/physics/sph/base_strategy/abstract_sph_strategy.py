from abc import ABC, abstractmethod
import numpy as np
from numba import cuda
from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters
from sim.src.main.physics.sph.base_strategy.kernels import collision_kernel_box
from sim.src.main.physics.sph.thread_organizer import ThreadOrganizer

MAX_NEIGHBOURS = 32


class AbstractSPHStrategy(ABC):
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.dt = 1 / self.params.n_particles
        self.old_state: SimulationState = None
        self.new_state: SimulationState = None
        self.thread_organizer = ThreadOrganizer()
        thread_setup = self.thread_organizer.organize(params.n_particles)
        self.grid_size = thread_setup[0]
        self.block_size = thread_setup[1]

    def compute_next_state(self, old_state: SimulationState) -> SimulationState:
        self.old_state = old_state
        self._initialize_computation()
        self._compute_density()
        self._compute_pressure()
        self._compute_viscosity()
        self._integrate()
        self._collide()
        self._finalize_computation()
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

    @abstractmethod
    def _integrate(self):
        pass

    @abstractmethod
    def _finalize_computation(self):
        pass

    def _collide(self):
        self.d_space_size = cuda.to_device(self.params.space_size)
        collision_kernel_box[self.grid_size, self.block_size](
            self.d_position,
            self.d_velocity,
            self.d_space_size
        )
        cuda.synchronize()

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
