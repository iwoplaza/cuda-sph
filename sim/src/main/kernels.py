from numpy import ndarray
from numba import cuda
import numpy as np
import math


@cuda.jit
def density_kernel(density: ndarray, position: ndarray, mass: np.float64, inf_radius: np.float64):
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
    mass: np.float64,
    inf_radius: np.float64,
    k: np.float64,
    ro_0: np.float64,
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
    mass: np.float64,
    inf_radius: np.float64,
    viscosity_const: np.float64,
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

@cuda.jit
def integrating_kernel(
    external_force: ndarray,
    pressure_term: ndarray,
    viscosity_term: ndarray,
    position: ndarray,
    velocity: np.float64,
    dt: np.float64,
    mass: np.float64
):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    i = block_width * block_idx + th_idx
    if i >= position.shape[0]:
        return
    
    # perform numerical integration with 'dt' timestep (in seconds)
    result_force = cuda.local.array(3, np.float64)
    for dim in range(3):
        result_force[dim] = (
            external_force[dim] +
            pressure_term[i][dim] +
            viscosity_term[i][dim]
            )
        velocity[i][dim] += result_force[dim] / mass * dt
        position[i][dim] += velocity[i][dim] * dt

if __name__ == "__main__":
    N_ELEMENTS = int(4)
    block_size = 128
    n_blocks = math.ceil(N_ELEMENTS / block_size)

    MASS = 1e-5
    K_CONST = 0.5
    RO_0_CONST = 0.056
    INF_RADIUS = 1e-4

    dt = 1/60

    external_force = np.random.random(3).astype("float64")

    dens = np.zeros(N_ELEMENTS).astype("float64")
    old_state = {
        "position": (
            np.random.random(3 * N_ELEMENTS)
            .reshape((N_ELEMENTS, 3))
            .astype("float64")
        ),
        "density": (
            np.random.random(N_ELEMENTS)
            .astype("float64")
        ),
        "velocity": (
            np.random.random(3 * N_ELEMENTS)
            .reshape((N_ELEMENTS, 3))
            .astype("float64")
        ),
        "pressure_term": (
            np.zeros(3 * N_ELEMENTS)
            .reshape((N_ELEMENTS, 3))
            .astype("float64")
        ),
        "viscosity_term": (
            np.zeros(3 * N_ELEMENTS)
            .reshape((N_ELEMENTS, 3))
            .astype("float64")
        )
    }
    
    # send to gpu
    d_density = cuda.to_device(old_state["density"])
    d_position = cuda.to_device(old_state["position"])
    d_velocity = cuda.to_device(old_state["velocity"])
    d_pressure_term = cuda.to_device(old_state["pressure_term"])
    d_viscosity_term = cuda.to_device(old_state["viscosity_term"])
    d_external_force = cuda.to_device(external_force)

    integrating_kernel[n_blocks, block_size](
    d_external_force,
    d_pressure_term,
    d_viscosity_term,
    d_position,
    d_velocity,
    dt,
    MASS)

    print(d_position.copy_to_host())
