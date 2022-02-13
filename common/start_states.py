import numpy as np
import config
from common.data_classes import SimulationParameters, SimulationState


def pouring(params: SimulationParameters) -> SimulationState:
    position = np.random \
        .random(params.n_particles * 3) \
        .reshape((params.n_particles, 3)) \
        .astype("float64")
    for i in range(params.n_particles):
        for dim in range(3):
            position[i][dim] *= params.space_size[dim] * 0.1
            position[i][dim] += params.space_size[dim] * 0.9

    velocity = np.random \
        .random(params.n_particles * 3) \
        .reshape((params.n_particles, 3)).astype("float64")
    for i in range(params.n_particles):
        velocity[i][0] *= 100.0
        velocity[i][0] += -500.0
        velocity[i][1] *= -10.0
        velocity[i][2] *= 150.0
        velocity[i][2] += -250.0

    density = np.zeros(params.n_particles).astype("float64")
    for i in range(params.n_particles):
        density[i] += 0.5
        density[i] *= config.RHO_0

    return SimulationState(position, velocity, density)
