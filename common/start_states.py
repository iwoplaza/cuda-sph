import random
import numpy as np
import config
from common.data_classes import SimulationParameters, SimulationState


def pouring(params: SimulationParameters) -> SimulationState:
    # put particles into one small cube in the upper corner of the space
    position = np.random \
        .random(params.n_particles * 3) \
        .reshape((params.n_particles, 3)) \
        .astype("float64")
    for i in range(params.n_particles):
        position[i][1] *= params.space_size[1] * 0.1
        # position[i][1] += params.space_size[1] * 0.6

        position[i][0] *= params.space_size[0] * 0.1
        # position[i][0] += params.space_size[0] * 0.9

        position[i][2] *= params.space_size[2] * 0.1

    # set their moving direction ate the space's diagonal (and randomize it a bit)
    base_velocity = [-29.0, 0.0, 29.0]
    offset_range = 13.0
    velocity = np.zeros(params.n_particles * 3) \
        .reshape((params.n_particles, 3)) \
        .astype("float64")
    for i in range(params.n_particles):
        for dim in range(3):
            velocity[i][dim] = base_velocity[dim] + ((random.random() - 0.5) * offset_range)

    # density is zero, due to the fact that it's always computed
    # based on distance between particles (we cannot arbitrarily set it)
    density = np.zeros(params.n_particles).astype("float64")

    return SimulationState(position, velocity, density)