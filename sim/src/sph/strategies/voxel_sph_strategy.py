import math
import logging
import numpy as np
from numba import cuda
import sim.src.sph.kernels.voxel_kernels as kernels
from common.data_classes import SimulationParameters
from config import MASS, INF_R, K, RHO_0, VISC, MAX_NEIGHBOURS, NEIGHBOURING_VOXELS_COUNT
from sim.src.sph.strategies.abstract_sph_strategy import AbstractSPHStrategy
import sim.src.sph.kernels.voxel_kernels as kernels
from numba import cuda
import numba.cuda.random as random
import numpy as np
import math
from sim.src.sph.strategies.abstract_sph_strategy import AbstractSPHStrategy
from sim.src.sph.kernels import voxel_kernels
from sim.src.sph.logging_utils import sph_stage_logger

logger = logging.getLogger(__name__)


class VoxelSPHStrategy(AbstractSPHStrategy):

    def __init__(self, params: SimulationParameters):
        super().__init__(params)

        self.neighbours = np.zeros((self.params.particle_count, MAX_NEIGHBOURS+1), dtype=np.int32)

    @sph_stage_logger(logger)
    def _initialize_computation(self):
        self.rng_states = random.create_xoroshiro128p_states(self.grid_size * self.block_size, seed=1)
        super()._send_arrays_to_gpu()
        self.__organize_voxels()
        self.__initialize_space_dim()
        # print(self.neighbours)
        # self.neighbours = np.zeros((self.params.n_particles, MAX_NEIGHBOURS + 1), dtype=np.int32)
        self.d_neighbours = cuda.to_device(self.neighbours)
        self.diagnostic_arr = -420 * np.ones(9)
        self.d_diagnostic_arr = cuda.to_device(self.diagnostic_arr)
        self.__compute_neighbours()
        self.neighbours = self.d_neighbours.copy_to_host()
        self.__inspect_neighbours()

    def __inspect_neighbours(self):
        self.diagnostic_arr = self.d_diagnostic_arr.copy_to_host()
        max_rows = np.max(np.abs(self.neighbours), axis=1)
        argmax = np.argmax(max_rows)
        assert max_rows[argmax] < self.params.particle_count, f"{max_rows[argmax]} {self.neighbours[argmax, :]}, {self.diagnostic_arr} {self.voxel_begin}"



    def __compute_neighbours(self):
        voxel_kernels.neighbours_kernel[self.grid_size, self.block_size](
            self.d_neighbours,
            self.d_position,
            self.d_voxel_size,
            self.d_space_dim,
            self.d_voxel_begin,
            self.d_voxel_particle_map,
            self.rng_states,
            self.d_diagnostic_arr
        )
        cuda.synchronize()


    @sph_stage_logger(logger)
    def _compute_density(self):
        voxel_kernels.density_kernel[self.grid_size, self.block_size](
            self.d_new_density,
            self.d_position,
            self.d_neighbours
        )
        cuda.synchronize()

    @sph_stage_logger(logger)
    def _compute_pressure(self):
        voxel_kernels.pressure_kernel[self.grid_size, self.block_size](
            self.d_new_pressure_term,
            self.d_new_density,
            self.d_position,
            self.d_neighbours
        )
        cuda.synchronize()

    @sph_stage_logger(logger)
    def _compute_viscosity(self):
        voxel_kernels.viscosity_kernel[self.grid_size, self.block_size](
            self.d_new_viscosity_term,
            self.d_new_density,
            self.d_position,
            self.d_velocity,
            self.d_neighbours

        )
        cuda.synchronize()

    @sph_stage_logger(logger)
    def __organize_voxels(self):
        # assign voxel indices to particles
        space_size = self.params.space_size
        voxel_size = self.params.voxel_size
        self.d_voxel_size = cuda.to_device(voxel_size)
        space_dims = np.asarray(
            [math.ceil(space_size[dim] / voxel_size[dim]) for dim in range(3)],
            dtype=np.int32
        )
        d_voxels = cuda.to_device(np.zeros(self.params.particle_count, dtype=np.int32))  # new buffer for voxel indices
        kernels.assign_voxels_to_particles_kernel[self.grid_size, self.block_size](
            d_voxels,
            self.d_position,
            cuda.to_device(np.asarray(self.params.voxel_size, np.float64)),
            cuda.to_device(space_dims),
        )
        self.voxels = d_voxels.copy_to_host()

        # create and sort (voxel_idx, particles_id) map
        self.voxel_particle_map = np.asarray(
            [(self.voxels[i], i) for i in range(self.params.particle_count)],
            dtype=[("voxel_id", np.int32), ("particle_id", np.int32)],
        )
        self.voxel_particle_map.sort(order="voxel_id")
        self.d_voxel_particle_map = cuda.to_device(self.voxel_particle_map)

        # create and populate voxel_begin array
        n_voxels = space_dims[0] * space_dims[1] * space_dims[2]
        self.voxel_begin = np.array([-1 for _ in range(n_voxels)], dtype=np.int32)
        self.__populate_voxel_begins()
        self.d_voxel_begin = cuda.to_device(self.voxel_begin)

    @sph_stage_logger(logger)
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

    @sph_stage_logger(logger)
    def __initialize_space_dim(self):
        self.space_dim = np.asarray(
            [np.int32(self.params.space_size[dim] / self.params.voxel_size[dim])
             for dim in range(3)],
            dtype=np.int32
        )
        self.d_space_dim = cuda.to_device(self.space_dim)
