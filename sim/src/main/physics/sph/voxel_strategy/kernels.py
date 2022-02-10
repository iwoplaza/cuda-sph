from sim.src.main.physics.sph.base_strategy import get_index
from numba import cuda
import numpy as np
from sim.src.main.physics.sph.base_strategy.abstract_sph_strategy import MAX_NEIGHBOURS


@cuda.jit(device=True)
def compute_3d_voxel_idx(voxel, idx, position, voxel_size):
    for dim in range(3):
        voxel[dim] = np.int32(position[idx][dim] / voxel_size[dim])


@cuda.jit(device=True)
def compute_1d_idx(voxel, space_dim):
    return voxel[0] + voxel[1] * space_dim[0] + voxel[2] * space_dim[0] * space_dim[1]


@cuda.jit(device=True)
def get_neighbours(
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map
):
    voxel = cuda.local.array(3, np.int32)
    compute_3d_voxel_idx(voxel, p_idx, position, voxel_size)

    voxel_neighbours = cuda.local.array(9, np.int32)
    i = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                neighbour_voxel_3d_idx = cuda.local.array(3, np.int32)
                neighbour_voxel_3d_idx[0] = voxel[0] + x
                neighbour_voxel_3d_idx[1] = voxel[1] + y
                neighbour_voxel_3d_idx[2] = voxel[2] + z
                voxel_neighbours[i] = compute_1d_idx(neighbour_voxel_3d_idx, space_dim)

    nei_idx = 0
    for i in voxel_neighbours:
        start = voxel_begin[i]
        if start == -1:
            continue
        next_voxel_idx = i + 1
        while voxel_begin[next_voxel_idx] == -1 and next_voxel_idx < len(voxel_begin):
            next_voxel_idx += 1
        end = (
            len(voxel_particle_map)
            if next_voxel_idx == len(voxel_begin)
            else voxel_begin[next_voxel_idx]
        )
        for map_entry in range(start, end + 1):
            neighbours[nei_idx] = voxel_particle_map[map_entry][1]
            nei_idx += 1
            if nei_idx >= len(neighbours):
                return nei_idx
    return nei_idx


@cuda.jit
def assign_voxels_to_particles_kernel(
        voxels: np.ndarray,
        position: np.ndarray,
        voxel_size: np.ndarray,
        space_dim: np.ndarray
):
    i = get_index()
    if i >= position.shape[0]:
        return

    # compute 3d index of a voxel
    voxel = cuda.local.array(3, np.int32)
    compute_3d_voxel_idx(i, position, voxel_size, voxel)

    # compute 1d index of a voxel and return
    # idx = x + y*w + z*w*d
    voxels[i] = compute_1d_idx(voxel, space_dim)


@cuda.jit
def density_kernel(
        result_density: np.ndarray,
        position: np.ndarray,
        voxel_begin,
        voxel_particle_map,
        voxel_size,
        space_dim,
        MASS: np.float64,
        INF_R: np.float64,
):
    i = get_index()
    if i >= position.shape[0]:
        return

    neighbours = cuda.local.array(MAX_NEIGHBOURS, np.int32)
    last_nei = get_neighbours(neighbours, i, position, voxel_size, space_dim, voxel_begin, voxel_particle_map)
    new_density = 0
    for j in neighbours[:last_nei]:
        if j == i:
            continue
        dist_norm = (
                (position[i][0] - position[j][0]) ** 2
                + (position[i][1] - position[j][1]) ** 2
                + (position[i][2] - position[j][2]) ** 2
        )
        new_density += MASS * (315 / 64 * np.pi * INF_R ** 9) * (INF_R ** 2 - dist_norm) ** 3
    result_density[i] = new_density
