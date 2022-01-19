from numpy import float64, ndarray
from numba import cuda
import numpy as np
import math


@cuda.jit
def density_kernel(
    result_density: ndarray,
    position: ndarray,
    MASS: float64,
    INF_R: float64
):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= result_density.shape[0]:
        return
    new_density = 0
    for j in range(result_density.shape[0]):
        if j == i:
            continue
        dist_norm = (
            (position[i][0] - position[j][0]) ** 2
            + (position[i][1] - position[j][1]) ** 2
            + (position[i][2] - position[j][2]) ** 2
        )
        new_density += (
            MASS
            * (315 / 64 * np.pi * INF_R ** 9)
            * (INF_R ** 2 - dist_norm) ** 3
        )
    result_density[i] = new_density


@cuda.jit
def pressure_kernel(
    result_pressure_term: ndarray,
    density: ndarray,
    position: ndarray,
    MASS: float64,
    INF_R: float64,
    K: float64,
    RHO_0: float64,
):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= density.shape[0]:
        return 
    new_pressure_term = cuda.local.array(3, np.double)
    for j in range(density.shape[0]):
        if j == i:
            continue
        dist = cuda.local.array(3, np.double)
        for dim in range(3):
            dist[dim] = position[i][dim] - position[j][dim]
        dist_norm = math.sqrt(
            (position[i][0] - position[j][0]) ** 2
            + (position[i][1] - position[j][1]) ** 2
            + (position[i][2] - position[j][2]) ** 2
        )
        w_grad = (
            (-45 / np.pi * INF_R ** 6)
            * (INF_R - dist_norm ** 2)
            / dist_norm
        )
        p_i = K * (density[i] - RHO_0)
        p_j = K * (density[j] - RHO_0)
        for dim in range(3):
            new_pressure_term[dim] += (
                dist[dim]
                * MASS
                * (p_i / density[i] ** 2 + p_j / density[j] ** 2)
                * w_grad
            )
    for dim in range(3):
        result_pressure_term[i][dim] = new_pressure_term[dim]


@cuda.jit
def viscosity_kernel(
    result_viscosity_term: ndarray,
    density: ndarray,
    position: ndarray,
    velocity: ndarray,
    MASS: float64,
    INF_R: float64,
    VISC: float64,
):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= density.shape[0]:
        return 
    new_viscosity_term = cuda.local.array(3, np.double)
    for j in range(density.shape[0]):
        if j == i:
            continue
        dist = cuda.local.array(3, np.double)
        velocity_diff = cuda.local.array(3, np.double)
        for dim in range(3):
            dist[dim] = position[i][dim] - position[j][dim]
            velocity_diff[dim] = velocity[j][dim] - velocity[i][dim]
        dist_norm = math.sqrt(
            (position[i][0] - position[j][0]) ** 2
            + (position[i][1] - position[j][1]) ** 2
            + (position[i][2] - position[j][2]) ** 2
        )
        w_laplacian = (45 / np.pi * INF_R ** 6) * (INF_R - dist_norm ** 2)
        for dim in range(3):
            new_viscosity_term[dim] += (
                MASS * velocity_diff[dim] / density[j] * w_laplacian
            )
    for dim in range(3):
        result_viscosity_term[i][dim] = (
            VISC * new_viscosity_term[dim] / density[i]
        )

@cuda.jit
def integrating_kernel(
    updated_position: ndarray,
    updated_velocity: ndarray,
    external_force: ndarray,
    pressure_term: ndarray,
    viscosity_term: ndarray,
    DT: float64,
    MASS: float64
):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= updated_position.shape[0]:
        return
    
    # perform numerical integration with 'dt' timestep (in seconds)
    result_force = cuda.local.array(3, np.float64)
    for dim in range(3):
        result_force[dim] = (
            external_force[dim] +
            pressure_term[i][dim] +
            viscosity_term[i][dim]
            )
        updated_velocity[i][dim] += result_force[dim] / MASS * DT
        updated_position[i][dim] += updated_velocity[i][dim] * DT
