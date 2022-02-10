from sim.src.main.physics.sph.base_strategy import get_index
from numba import cuda
import numpy as np
import math


@cuda.jit
def density_kernel(
        result_density: np.ndarray,
        position: np.ndarray,
        MASS: np.float64,
        INF_R: np.float64
):
    i = get_index()
    if i >= position.shape[0]:
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
        result_pressure_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        MASS: np.float64,
        INF_R: np.float64,
        K: np.float64,
        RHO_0: np.float64,
):
    i = get_index()
    if i >= position.shape[0]:
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
        result_viscosity_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        MASS: np.float64,
        INF_R: np.float64,
        VISC: np.float64,
):
    i = get_index()
    if i >= position.shape[0]:
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
