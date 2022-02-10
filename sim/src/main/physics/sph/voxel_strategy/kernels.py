from sim.src.main.physics.constants import INF_R
from sim.src.main.physics.sph.base_strategy import get_index
from numba import cuda
import numpy as np
from math import sqrt
from sim.src.main.physics.sph.base_strategy.abstract_sph_strategy import MAX_NEIGHBOURS


@cuda.jit(device=True)
def compute_3d_voxel_idx(voxel, idx, position, voxel_size):
    for dim in range(3):
        voxel[dim] = np.int32(position[idx][dim] / voxel_size[dim])


@cuda.jit(device=True)
def compute_1d_idx(voxel, space_dim):
    return voxel[0] + voxel[1] * space_dim[0] + voxel[2] * space_dim[0] * space_dim[1]


@cuda.jit(device=True)
def are_neighbours(p1, p2, positions):
    return sqrt(
        (positions[p1][0] - positions[p2][0]) ** 2 +
        (positions[p1][1] - positions[p2][1]) ** 2 +
        (positions[p1][2] - positions[p2][2]) ** 2
    ) <= INF_R


@cuda.jit(device=True)
def get_neighbours(
        voxel,
        neigh_voxels,
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map
):
    compute_3d_voxel_idx(voxel, p_idx, position, voxel_size)
    # find all neighbouring voxels
    i = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                neigh_voxel = cuda.local.array(3, np.int32)
                neigh_voxel[0] = voxel[0] + dx
                neigh_voxel[1] = voxel[1] + dy
                neigh_voxel[2] = voxel[2] + dz
                idx_1d = compute_1d_idx(neigh_voxel, space_dim)
                if neigh_voxel[0] >= 0 and neigh_voxel[1] >= 0 and neigh_voxel[2] >= 0:
                    neigh_voxels[i] = idx_1d
                    i += 1

    neigh_idx = 0
    # for each voxel...
    for i in neigh_voxels:
        # ...find start and end of it in map...
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
        # ... and add up to MAX_NEIGHBOURS particles from these voxels
        for map_entry in range(start, end):
            potential_neigh = voxel_particle_map[map_entry][1]
            if are_neighbours(p_idx, potential_neigh, position):
                neighbours[neigh_idx] = potential_neigh
                neigh_idx += 1
                if neigh_idx >= len(neighbours):
                    return neigh_idx
    return neigh_idx


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
    compute_3d_voxel_idx(voxel, i, position, voxel_size)

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
    neigh_count = get_neighbours(neighbours, i, position, voxel_size, space_dim, voxel_begin, voxel_particle_map)
    new_density = 0
    for j in neighbours[:neigh_count]:
        if j == i:
            continue
        dist_norm = (
                (position[i][0] - position[j][0]) ** 2
                + (position[i][1] - position[j][1]) ** 2
                + (position[i][2] - position[j][2]) ** 2
        )
        new_density += MASS * (315 / 64 * np.pi * INF_R ** 9) * (INF_R ** 2 - dist_norm) ** 3
    result_density[i] = new_density
