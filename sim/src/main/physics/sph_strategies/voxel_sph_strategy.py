from common.main.data_classes.simulation_data_classes import SimulationState
from sim.src.main.physics.sph_strategies.abstract_sph_strategy import AbstractSPHStrategy
import sim.src.main.physics.kernels as kernels
import sim.src.main.physics.constants as constants
from numba import cuda
import numpy as np
import math

# TODO: Implement voxel organization on gpu


class VoxelSPHStrategy(AbstractSPHStrategy):
    @property
    def next_state(self) -> SimulationState:
        return SimulationState(None, None, None, None)

    def _initialize_next_state(self, old_state):
        self.__state = old_state
        self.d_position = cuda.to_device(self.__state.position)
        self.d_velocity = cuda.to_device(self.__state.velocity)
        self.d_external_force = cuda.to_device(super().params.external_force)

        # b) arrays to be used during computations
        self.d_new_pressure_term = cuda.to_device(
            np.zeros((super().params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_viscosity_term = cuda.to_device(
            np.zeros((super().params.n_particles, 3), dtype=np.float64)
        )
        self.d_new_density = cuda.to_device(np.zeros(super().params.n_particles, dtype=np.float64))
        self.threads_per_grid: int = 64
        self.grids_per_block: int = math.ceil(super().params.n_particles / self.threads_per_grid)

    def _organize_voxels(self, state: SimulationState):
        # assign voxel indices to particles
        space_size = self.params.space_size
        voxel_size = self.params.voxel_size
        space_dims = np.asarray(
            [math.ceil(space_size[dim] / voxel_size[dim]) for dim in range(3)], dtype=np.int32
        )
        self.d_new_voxels = cuda.to_device(
            np.zeros(super().n_particles, dtype=np.int32)
        )  # new buffer for voxel indices
        kernels.assign_voxels_to_particles_kernel[self.threads_per_grid, self.grids_per_block](
            self.d_new_voxels,
            self.d_position,
            np.asarray(self.params.voxel_size, np.float64),
            cuda.to_device(space_dims),
        )
        self.voxels = self.d_new_voxels.copy_to_host()

        # create and sort (voxel_idx, particles_id) list
        self.voxel_particle_map = np.asarray(
            [(self.voxels[i], i) for i in range(super().n_particles)],
            dtype=[("voxel_id", np.int32), ("particle_id", np.int32)],
        )
        self.voxel_particle_map.sort(order="voxel_id")

        # create and populate voxel_begin array by linearly iterating over voxel_map assign a beginning to each voxel
        n_voxels = space_dims[0] * space_dims[1] * space_dims[2]
        self.voxel_begin = np.array([-1 for _ in range(n_voxels)], dtype=np.int32)

        # TODO: fix this, it has some error
        self.__populate_voxel_begins()
        self.d_voxel_particle_map = cuda.to_device(self.voxel_particle_map)
        self.d_voxel_begin = cuda.to_device(self.voxel_begin)

    def __populate_voxel_begins(self):
        map_idx = 0
        for voxel_idx in range(len(self.voxel_begin)):
            while self.voxel_particle_map[map_idx][0] < voxel_idx:
                map_idx += 1
                if map_idx >= len(self.voxel_particle_map):
                    return
            if self.voxel_particle_map[map_idx][0] == voxel_idx:
                self.voxel_begin[voxel_idx] = map_idx
        return

    def _compute_density(self):
        pass

    def _compute_pressure(self):
        pass

    def _compute_viscosity(self):
        pass

    def _integrate(self):
        pass
