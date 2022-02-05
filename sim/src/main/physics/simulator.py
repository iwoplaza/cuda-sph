from numpy import zeros, int32, float64
from math import ceil
from numba import cuda
from common.main.data_classes.simulation_data_classes import SimulationState
import sim.src.main.physics.constants as constants
import sim.src.main.physics.kernels as kernels


class Simulator:
    def __init__(self, params) -> None:
        self.params = params
        self.dt = 1.0 / params.fps

    def compute_next_state(self, old_state) -> SimulationState:
        self.__organize_voxels(old_state)

        return self.__compute_physics(old_state)
    
    def __compute_physics(self, old_state) -> SimulationState:
        N: int = self.params.n_particles

        # send data to gpu
        # a) data to be updated or populated
        d_position = cuda.to_device(old_state.position)
        d_velocity = cuda.to_device(old_state.velocity)
        d_external_force = cuda.to_device(self.params.external_force)

        # b) arrays to be used during computations
        d_new_pressure_term = cuda.to_device(zeros((N, 3), dtype=float64))
        d_new_viscosity_term = cuda.to_device(zeros((N, 3), dtype=float64))
        d_new_density = cuda.to_device(zeros(N, dtype=float64))


        # compute block size and grid size
        # TODO: compute it using gpu detection for optimization
        threads_per_grid: int = 64
        grids_per_block: int = ceil(self.params.n_particles / threads_per_grid)

        # run kernels
        kernels.density_kernel[grids_per_block, threads_per_grid](
            d_new_density, d_position, constants.MASS, constants.INF_R
        )
        cuda.synchronize()

        kernels.pressure_kernel[grids_per_block, threads_per_grid](
            d_new_pressure_term, d_new_density, d_position,
            constants.MASS, constants.INF_R, constants.K, constants.RHO_0
        )

        kernels.viscosity_kernel[grids_per_block, threads_per_grid](
            d_new_viscosity_term, d_new_density, d_position, d_velocity,
            constants.MASS, constants.INF_R, constants.VISC
        )
        cuda.synchronize()

        kernels.integrating_kernel[grids_per_block, threads_per_grid](
            d_position, d_velocity, d_external_force, d_new_pressure_term,
            d_new_viscosity_term, self.dt, constants.MASS
        )
        cuda.synchronize()

        return SimulationState(
            d_position.copy_to_host(),
            d_velocity.copy_to_host(),
            d_new_density.copy_to_host(),
            zeros(N, dtype=int32)
        )

    def __organize_voxels(self, old_state) -> None:
        # :)
        pass