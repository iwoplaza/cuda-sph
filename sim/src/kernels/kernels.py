from copyreg import dispatch_table

# from numpy.typing import ndarray
from numpy import ndarray
from numba import cuda
import numpy as np
import cupy as cp
import math


@cuda.jit
def density_kernel(density: ndarray, position: ndarray, mass: float, inf_radius: float):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= density.shape[0]:
        return
    new_density = 0
    for j in range(density.shape[0]):
        if j == i:
            continue
        dist_norm = (
            (position[i][0] - position[j][0]) ** 2
            + (position[i][1] - position[j][1]) ** 2
            + (position[i][2] - position[j][2]) ** 2
        )
        new_density += (
            mass
            * (315 / 64 * np.pi * inf_radius ** 9)
            * (inf_radius ** 2 - dist_norm) ** 3
        )
    density[i] = new_density


@cuda.jit
def pressure_kernel(
    density: ndarray,
    position: ndarray,
    pressure_term: ndarray,
    mass: float,
    inf_radius: float,
    k: float,
    ro_0: float,
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
            (-45 / np.pi * inf_radius ** 6)
            * (inf_radius - dist_norm ** 2)
            / dist_norm
        )
        p_i = k * (density[i] - ro_0)
        p_j = k * (density[j] - ro_0)
        for dim in range(3):
            new_pressure_term[dim] += (
                dist[dim]
                * mass
                * (p_i / density[i] ** 2 + p_j / density[j] ** 2)
                * w_grad
            )
    for dim in range(3):
        pressure_term[i][dim] = new_pressure_term[dim]


@cuda.jit
def viscosity_kernel(
    density: ndarray,
    position: ndarray,
    velocity: ndarray,
    viscosity_term: ndarray,
    mass: float,
    inf_radius: float,
    viscosity_const: float,
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
        w_laplacian = (45 / np.pi * inf_radius ** 6) * (inf_radius - dist_norm ** 2)
        for dim in range(3):
            new_viscosity_term[dim] += (
                mass * velocity_diff[dim] / density[j] * w_laplacian
            )
    for dim in range(3):
        viscosity_term[i][dim] = (
            viscosity_const * new_viscosity_term[dim] / density[i]
        )


def integrating_kernel(
    density: ndarray, postion: ndarray, mass: float, inf_radius: float
):
    pass


if __name__ == "__main__":
    N_ELEMENTS = int(3e4)
    GRID_SIZE = 128
    dens = np.zeros(N_ELEMENTS).astype("float64")
    pos = (
        np.random.randint(0, 127, 3 * N_ELEMENTS)
        .reshape((N_ELEMENTS, 3))
        .astype("float32")
    )
    pos2 = (
        np.random.randint(0, 127, 3 * N_ELEMENTS)
        .reshape((N_ELEMENTS, 3))
        .astype("float32")
    )
    pos3 = (
        np.zeros(3 * N_ELEMENTS)
        .reshape((N_ELEMENTS, 3))
        .astype("float32")
    )
    d_dens = cuda.to_device(dens)
    d_pos = cuda.to_device(pos)
    d_pos2 = cuda.to_device(pos2)
    d_pos3 = cuda.to_device(pos3)

    print(pos)
    pressure_kernel[N_ELEMENTS // GRID_SIZE + 10, GRID_SIZE](
        d_dens, d_pos, d_pos3, 0.1,  0.1, 0.1, 1000
    )
    print(np.sum(d_pos3.copy_to_host().reshape))
