from config import INF_R_2, K, RHO_0, MASS, VISC
from sim.src.sph.kernels.base_kernels import get_index, compute_w, norm, norm_squared, compute_grad_w, compute_lap_w
from numba import cuda
import numpy as np


@cuda.jit
def density_kernel(
        result_density: np.ndarray,
        position: np.ndarray,
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_density = 0
    for j in range(result_density.shape[0]):
        if norm_squared(position[i], position[j]) > INF_R_2:
            continue
        new_density += compute_w(position[i], position[j])
    result_density[i] = new_density * MASS


@cuda.jit
def pressure_kernel(
        result_pressure_term: np.ndarray,
        density: np.ndarray,
        position: np.ndarray,
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_pressure_term = cuda.local.array(3, np.float64)
    for dim in range(3):
        new_pressure_term[dim] = 0.0

    for j in range(density.shape[0]):
        if j == i or norm_squared(position[i], position[j]) > INF_R_2:
            continue

        p_i = K * (density[i] - RHO_0)
        p_j = K * (density[j] - RHO_0)
        # factor = p_i / density[i] ** 2 + p_j / density[j] ** 2
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
):
    i = get_index()
    if i >= position.shape[0]:
        return

    new_viscosity_term = cuda.local.array(3, np.double)
    for dim in range(3):
        new_viscosity_term[dim] = 0.0

    for j in range(density.shape[0]):
        if j == i or norm_squared(position[i], position[j]) > INF_R_2:
            continue

        lap_w = compute_lap_w(position[i], position[j])
        for dim in range(3):
            visc_term = (velocity[j][dim] - velocity[i][dim]) / density[j] * lap_w * VISC
            new_viscosity_term[dim] += visc_term * MASS

    for dim in range(3):
        result_viscosity_term[i][dim] = new_viscosity_term[dim]
