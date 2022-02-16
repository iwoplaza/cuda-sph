import random
import numpy as np
from numba import cuda
from common.data_classes import SimulationParameters, SimulationState, Pipe
from sim.src.sph import thread_layout
from sim.src.sph.kernels.base_kernels import spawn_particles_inside_pipe_kernel


def pouring(params: SimulationParameters) -> SimulationState:
    position = np.random \
        .random(params.particle_count * 3) \
        .reshape((params.particle_count, 3)) \
        .astype("float64")
    for i in range(params.particle_count):
        position[i][0] *= params.space_size[0]
        position[i][1] *= params.space_size[1]
        position[i][2] *= params.space_size[2]

    base_velocity = [-16.0, 0.0, 16.0]
    offset_range = 13.0
    velocity = np.zeros(params.particle_count * 3) \
        .reshape((params.particle_count, 3)) \
        .astype("float64")
    for i in range(params.particle_count):
        for dim in range(3):
            velocity[i][dim] = base_velocity[dim] + ((random.random() - 0.5) * offset_range)

    density = np.zeros(params.particle_count).astype("float64")

    return SimulationState(position, velocity, density)


def inside_pipe(params: SimulationParameters, pipe: Pipe) -> SimulationState:
    position = np.zeros((params.particle_count, 3))\
        .astype(np.float64)
    velocity = np.zeros((params.particle_count, 3))\
        .astype(np.float64)
    grid_size, block_size = thread_layout.organize(params.particle_count)
    d_position = cuda.to_device(position)
    d_velocity = cuda.to_device(velocity)
    d_pipe = cuda.to_device(pipe.to_numpy())
    rng_states = random.create_xoroshiro128p_states(grid_size * block_size, seed=17349)
    spawn_particles_inside_pipe_kernel[grid_size, block_size]\
        (d_position, d_velocity, d_pipe, rng_states)
    cuda.synchronize()
    position = d_position.copy_to_host()
    velocity = d_velocity.copy_to_host()
    density = np.zeros((params.particle_count, 3)).astype(np.float64)
    return SimulationState(
        position,
        velocity,
        density
    )
