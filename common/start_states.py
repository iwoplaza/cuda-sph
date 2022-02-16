import random
import numpy as np
from numba import cuda
import numba.cuda.random as cuda_rand
from common.data_classes import SimulationParameters, SimulationState, Pipe
from sim.src.sph import thread_layout
from sim.src.sph.kernels.base_kernels import spawn_particles_inside_pipe_kernel


def pouring(params: SimulationParameters) -> SimulationState:
    # put particles into one small cube in the upper corner of the space
    position = np.random \
        .random(params.n_particles * 3) \
        .reshape((params.n_particles, 3)) \
        .astype("float64")
    for i in range(params.n_particles):
        # position[i][1] *= params.space_size[1] * 0.1
        # position[i][1] += params.space_size[1] * 0.6
        position[i][1] *= params.space_size[1]

        # position[i][0] *= params.space_size[0] * 0.1
        # position[i][0] += params.space_size[0] * 0.9
        position[i][0] *= params.space_size[0]

        # position[i][2] *= params.space_size[2] * 0.1
        position[i][2] *= params.space_size[2]


    # set their moving direction at the space's diagonal (and randomize it a bit)
    base_velocity = [-16.0, 0.0, 16.0]
    offset_range = 13.0
    velocity = np.zeros(params.n_particles * 3) \
        .reshape((params.n_particles, 3)) \
        .astype("float64")
    for i in range(params.n_particles):
        for dim in range(3):
            # velocity[i][dim] = base_velocity[dim] + ((random.random() - 0.5) * offset_range)
            velocity[i][dim] = 0.0

    # density is zero, due to the fact that it's always computed
    # based on distance between particles (we cannot arbitrarily set it)
    density = np.zeros(params.n_particles).astype("float64")

    return SimulationState(position, velocity, density)


def inside_pipe(params: SimulationParameters, pipe: Pipe) -> SimulationState:
    position = np.random.rand(params.n_particles, 3).astype(np.float64)*params.space_size[0]
    velocity = np.zeros((params.n_particles, 3), dtype=np.float64)
    grid_size, block_size = thread_layout.organize(params.n_particles)
    d_position = cuda.to_device(position)
    d_pipe = cuda.to_device(pipe.to_numpy())
    rng_states = cuda_rand.create_xoroshiro128p_states(grid_size * block_size, seed=17349)
    spawn_particles_inside_pipe_kernel[grid_size, block_size]\
        (d_position, d_pipe, rng_states)
    cuda.synchronize()
    position = d_position.copy_to_host()
    density = np.zeros((params.n_particles, 3)).astype(np.float64)
    return SimulationState(
        position,
        velocity,
        density
    )
