import math
from numba import cuda
import numba.cuda.random as random
import numpy as np
from math import sqrt, floor
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
def neigh_voxels_1d_indices(neighbours, voxel_coord, space_dim):
    valid_voxels_count = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                neigh_voxel = cuda.local.array(3, np.int32)
                neigh_voxel[0] = voxel_coord[0] + dx
                neigh_voxel[1] = voxel_coord[1] + dy
                neigh_voxel[2] = voxel_coord[2] + dz
                # discard those not in space size
                is_in = True
                for dim in range(3):
                    if neigh_voxel[dim] < 0 or neigh_voxel[dim] >= space_dim[dim]:
                        is_in = False
                        break
                if is_in:
                    neighbours[valid_voxels_count] = compute_1d_idx(neigh_voxel, space_dim)
                    valid_voxels_count += 1
    return valid_voxels_count

@cuda.jit(device=True, debug=True)
def random_samples(samples, n_samples, minn, maxx, rng_states, idx):
    """fill samples array with random intagers from [minn, maxx) range"""
    n = n_samples if len(samples) > n_samples else len(samples)
    for i in range(n):
        rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)
        scaled_rand = rand * np.float32(maxx - minn)
        samples[i] = np.int32(scaled_rand + minn)
        assert samples[i] >= minn 
        assert samples[i] < maxx
    return n


@cuda.jit(device=True)
def get_voxel_id_from_acc(accum_particle_count, particle_idx) -> int:
    nei_voxel_idx = 0 
    # [10, 12, 12, 12, 15], 10 -> 1, 12 -> 4, 11 -> 1
    while accum_particle_count[nei_voxel_idx] <= particle_idx and nei_voxel_idx < len(accum_particle_count):
        nei_voxel_idx += 1
    return nei_voxel_idx


@cuda.jit(device=True)
def get_neighbours(
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map,
        rng_states,
        diagnostic_arr
):
    max_voxel_id = space_dim[0] * space_dim[1] * space_dim[2]


    # find out in which voxel we are
    voxel = cuda.local.array(3, np.int32)
    compute_3d_voxel_idx(voxel, p_idx, position, voxel_size)

    # find all neighbouring voxels
    nei_voxels_indices = cuda.local.array(NEIGHBOURING_VOXELS_COUNT, np.int32)
    nei_voxels_count = neigh_voxels_1d_indices(nei_voxels_indices, voxel, space_dim)

    # to all neighbouring voxels assign number of particles in it
    # create accumulated array as well
    particle_count_per_nei_voxel = cuda.local.array(NEIGHBOURING_VOXELS_COUNT, np.int32)
    accum_particle_count = cuda.local.array(NEIGHBOURING_VOXELS_COUNT, np.int32)
    total_neigh_count = 0
    nei_idx = 0
    for voxel_id in nei_voxels_indices[:nei_voxels_count]:
        if not (0 < voxel_id < max_voxel_id):
            return -1
        if voxel_begin[voxel_id] != -1:
            # get indices range, in which the particles of that voxel are located
            start = voxel_begin[voxel_id]
            next_voxel_idx = voxel_id + 1
            while voxel_begin[next_voxel_idx] == -1 and next_voxel_idx < len(voxel_begin):
                next_voxel_idx += 1
            if next_voxel_idx == len(voxel_begin):
                end = len(voxel_particle_map)
            else:
                end = voxel_begin[next_voxel_idx]
        else:
            end = 0
            start = 0

        particle_count_in_this_voxel = end - start
        total_neigh_count += particle_count_in_this_voxel
        accum_particle_count[nei_idx] = total_neigh_count
        particle_count_per_nei_voxel[nei_idx] = particle_count_in_this_voxel
        nei_idx += 1

    # get MAX_NEIGHBOURS random neighbours
    samples_size = np.int32(min(total_neigh_count, MAX_NEIGHBOURS))
    samples = cuda.local.array(MAX_NEIGHBOURS, np.int32)
    actual_size = random_samples\
        (samples, samples_size, 0, total_neigh_count,
         rng_states, p_idx)  # actual_size should be the same as samples_size in this case
    i = 0
    for nei_idx in samples[:actual_size]:
        if nei_idx == p_idx:
            continue
        nei_voxel_idx = get_voxel_id_from_acc(accum_particle_count, nei_idx)
        nei_voxel_start = voxel_begin[nei_voxels_indices[nei_voxel_idx]]

        # voxel_particles_num  = particle_count_per_nei_voxel[nei_voxel_idx]
        if nei_voxel_idx == 0:
            in_voxel_offset = nei_idx
        else:
            in_voxel_offset = nei_idx - accum_particle_count[nei_voxel_idx - 1]

        if nei_voxel_start + in_voxel_offset >= voxel_particle_map.shape[0]:
            diagnostic_arr[0] = nei_idx
            diagnostic_arr[1] = nei_voxel_idx
            diagnostic_arr[2] = nei_voxel_start
            diagnostic_arr[3] = in_voxel_offset
            diagnostic_arr[4] = nei_voxels_indices[nei_voxel_idx]
            diagnostic_arr[5] = accum_particle_count[nei_voxel_idx - 1]
            diagnostic_arr[6] = accum_particle_count[nei_voxel_idx]
            diagnostic_arr[7] = total_neigh_count
            diagnostic_arr[8] = particle_count_per_nei_voxel[nei_voxel_idx]
            neighbours[i] = -4200000000000000
        else:
            neighbours[i] = voxel_particle_map[nei_voxel_start + in_voxel_offset][1]
            if p_idx == 0:
                diagnostic_arr[0] = neighbours[i]
            i += 1
            if i >= MAX_NEIGHBOURS:
                return i
    return actual_size


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

@cuda.jit(device=True)
def fill_foo_array(foo, neighbours, neigh_count):
    foo[0] = neigh_count
    for i in range(MAX_NEIGHBOURS):
        foo[i+1] = neighbours[i]

@cuda.jit
def neighbours_kernel(
        neighbours_indices,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map,
        rng_states,
        diagnostic_arr
):
    i = get_index()
    if i >= position.shape[0]:
        return
    
    for j in range(MAX_NEIGHBOURS):
        neighbours_indices[i][j] = -10
    neigh_count = get_neighbours(
        neighbours_indices[i], i, position, voxel_size,
        space_dim, voxel_begin, voxel_particle_map, rng_states, diagnostic_arr
    )
    if i == 0:
        diagnostic_arr[1] = neigh_count

@cuda.jit
def density_kernel(
        result_density: np.ndarray,
        position: np.ndarray,
        neighbours
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_density = 0.0
    for j in neighbours[i]:
        if j < 0:
            break
        new_density += compute_w(position[i], position[j])
    result_density[i] = new_density * MASS


@cuda.jit
def pressure_kernel(
        result_pressure_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        neighbours
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_pressure_term = cuda.local.array(3, np.float64)
    for dim in range(3):
        new_pressure_term[dim] = 0.0

    for j in neighbours[i]:
        if j == i:
            continue
        if j < 0:
            break
        p_i = K * (density[i] - RHO_0)
        p_j = K * (density[j] - RHO_0)
        factor = (p_i + p_j) / (2 * density[j])
        grad_w = cuda.local.array(3, np.float64)
        compute_grad_w(position[i], position[j], grad_w)

        for dim in range(3):
            new_pressure_term[dim] += factor * grad_w[dim]

    for dim in range(3):
        result_pressure_term[i][dim] = new_pressure_term[dim] * MASS


@cuda.jit
def viscosity_kernel(
        result_viscosity_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        neighbours
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_viscosity_term = cuda.local.array(3, np.float64)
    for dim in range(3):
        new_viscosity_term[dim] = 0.0

    for j in neighbours[i]:
        if j == i:
            continue
        if j < 0:
            break
        lap_w = compute_lap_w(position[i], position[j])
        for dim in range(3):
            visc_term = (velocity[j][dim] - velocity[i][dim]) / density[j] * lap_w * VISC
            new_viscosity_term[dim] += visc_term * MASS

    for dim in range(3):
        result_viscosity_term[i][dim] = new_viscosity_term[dim]
