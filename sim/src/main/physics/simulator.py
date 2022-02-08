import math

import numpy as np
from numpy import zeros, int32, float64, ndarray
from math import ceil
from numba import cuda
from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters
import sim.src.main.physics.constants as constants
import sim.src.main.physics.kernels as kernels


class Simulator:
    def __init__(self, params) -> None:
        self.params: SimulationParameters = params
        self.dt = 1.0 / params.fps
        self.voxel_particle_map: ndarray
        self.voxel_begin: ndarray

    def compute_next_state(self, old_state: SimulationState) -> SimulationState:
        self.__organize_voxels(old_state)

        return self.__compute_physics(old_state)

    def __compute_physics(self, old_state: SimulationState) -> SimulationState:
        n_particles: int = self.params.n_particles

        # send data to gpu
        # a) data to be updated or populated
        d_position = cuda.to_device(old_state.position)
        d_velocity = cuda.to_device(old_state.velocity)
        d_external_force = cuda.to_device(self.params.external_force)

        # b) arrays to be used during computations
        d_new_pressure_term = cuda.to_device(zeros((n_particles, 3), dtype=float64))
        d_new_viscosity_term = cuda.to_device(zeros((n_particles, 3), dtype=float64))
        d_new_density = cuda.to_device(zeros(n_particles, dtype=float64))

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
            old_state.voxel
        )

    def __organize_voxels(self, state: SimulationState) -> None:
        n_particles: int = self.params.n_particles

        threads_per_grid: int = 64
        grids_per_block: int = ceil(self.params.n_particles / threads_per_grid)

        # assign voxel indices to particles
        space_size = self.params.space_size
        voxel_size = self.params.voxel_size
        space_dims = np.asarray([math.ceil(space_size[dim] / voxel_size[dim]) for dim in range(3)], dtype=int32)
        d_position = cuda.to_device(state.position)
        d_new_voxels = cuda.to_device(zeros(n_particles, dtype=int32))  # new buffer for voxel indices
        kernels.assign_voxels_to_particles_kernel[threads_per_grid, grids_per_block] \
            (d_new_voxels, d_position, np.asarray(self.params.voxel_size, float64), cuda.to_device(space_dims))
        state.voxel = d_new_voxels.copy_to_host()

        # create and sort (voxel_idx, particles_id) list
        self.voxel_particle_map = np.asarray([(state.voxel[i], i) for i in range(n_particles)],
                               dtype=[('voxel_id', int32), ('particle_id', int32)])
        self.voxel_particle_map.sort(order='voxel_id')

        # create and populate voxel_begin array by linearly iterating over voxel_map assign a beginning to each voxel
        n_voxels = space_dims[0] * space_dims[1] * space_dims[2]
        self.voxel_begin = np.array([-1 for _ in range(n_voxels)], dtype=int32)
        map_idx = 0
        # TODO: fix this, it has some error
        for voxel_idx in range(n_voxels):
            if self.voxel_particle_map[map_idx][0] == voxel_idx:
                self.voxel_begin[voxel_idx] = map_idx
            while self.voxel_particle_map[map_idx][0] < voxel_idx:
                map_idx += 1


