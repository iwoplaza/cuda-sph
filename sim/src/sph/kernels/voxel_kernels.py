import math
from numba import cuda
import numpy as np
from math import sqrt
from config import INF_R, NEIGHBOURING_VOXELS_COUNT, MAX_NEIGHBOURS, MASS, K, RHO_0, VISC
from sim.src.sph.kernels.base_kernels import get_index, compute_w, compute_grad_w, compute_lap_w


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
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map
):
    # find out in which voxel we are
    voxel = cuda.local.array(3, np.int32)
    compute_3d_voxel_idx(voxel, p_idx, position, voxel_size)

    # find all neighbouring voxels
    neigh_voxels = cuda.local.array(NEIGHBOURING_VOXELS_COUNT, np.int32)
    i = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                neigh_voxel = cuda.local.array(3, np.int32)
                neigh_voxel[0] = voxel[0] + dx
                neigh_voxel[1] = voxel[1] + dy
                neigh_voxel[2] = voxel[2] + dz
                # discard those not in space size
                is_in = True
                for dim in range(3):
                    if neigh_voxel[dim] < 0 or neigh_voxel[dim] >= space_dim[dim]:
                        is_in = False
                        break
                if is_in:
                    neigh_voxels[i] = compute_1d_idx(neigh_voxel, space_dim)
                    i += 1
    # for each neighbouring voxel...
    neigh_idx = 0
    for i in neigh_voxels:
        if i == -1 or voxel_begin[i] == -1:
            continue
        # ...find start and end of it in map...
        start = voxel_begin[i]
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
                if neigh_idx >= MAX_NEIGHBOURS:
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
):
    i = get_index()
    if i >= position.shape[0]:
        return

    neighbours = cuda.local.array(MAX_NEIGHBOURS, np.int32)
    neigh_count = get_neighbours(
        neighbours, i, position, voxel_size,
        space_dim, voxel_begin, voxel_particle_map
    )

    new_density = 0.0
    for j in neighbours[:neigh_count]:
        if j == i:
            continue
        new_density += compute_w(position[i], position[j])
    result_density[i] = new_density * MASS


@cuda.jit
def pressure_kernel(
        result_pressure_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        voxel_begin,
        voxel_particle_map,
        voxel_size,
        space_dim,
):
    i = get_index()
    if i >= position.shape[0]:
        return

    neighbours = cuda.local.array(MAX_NEIGHBOURS, np.int32)
    neigh_count = get_neighbours(
        neighbours, i, position, voxel_size,
        space_dim, voxel_begin, voxel_particle_map
    )

    new_pressure_term = cuda.local.array(3, np.float64)
    for dim in range(3):
        new_pressure_term[dim] = 0.0

    for j in neighbours[:neigh_count]:
        if j == i:
            continue
        p_i = K * (density[i] - RHO_0)
        p_j = K * (density[j] - RHO_0)
        factor = p_i / density[i] ** 2 + p_j / density[j] ** 2
        grad_w = cuda.local.array(3, np.float64)
        compute_grad_w(position[i], position[j], grad_w)
        for dim in range(3):
            new_pressure_term[dim] += factor * grad_w[dim]

    for dim in range(3):
        result_pressure_term[i][dim] = new_pressure_term[dim]


@cuda.jit
def viscosity_kernel(
        result_viscosity_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        voxel_begin,
        voxel_particle_map,
        voxel_size,
        space_dim,
):
    i = get_index()
    if i >= position.shape[0]:
        return

    neighbours = cuda.local.array(MAX_NEIGHBOURS, np.int32)
    neigh_count = get_neighbours(
        neighbours, i, position, voxel_size,
        space_dim, voxel_begin, voxel_particle_map
    )

    new_viscosity_term = cuda.local.array(3, np.double)
    for dim in range(3):
        new_viscosity_term[dim] = 0.0

    for j in neighbours[:neigh_count]:
        if j == i:
            continue
        visc_term_j = cuda.local.array(3, np.float64)
        lap_w = compute_lap_w(position[i], position[j])
        for dim in range(3):
            visc_term_j[dim] = (velocity[j][dim] - velocity[i][dim]) / \
                               density[j] * lap_w
            new_viscosity_term[dim] += visc_term_j[dim] / density[i]

    for dim in range(3):
        result_viscosity_term[i][dim] = new_viscosity_term[dim] * MASS * VISC
