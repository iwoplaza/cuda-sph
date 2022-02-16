"""
    This module has to be located in the project root directory.
    This module is really important, many modules depend on that.
"""


import math
import os
from enum import Enum, auto
from random import random
import numpy as np
from common.data_classes import SimulationState, SimulationParameters, Pipe
from common.pipe_builder import PipeBuilder


class SimMode(Enum):
    BOX = auto()
    PIPE = auto()


class SimStrategy(Enum):
    NAIVE = auto()
    VOXEL = auto()


SIM_MODE: SimMode = SimMode.PIPE
SIM_STRATEGY: SimStrategy = SimStrategy.NAIVE


DURATION = 10
FPS = 20
PARTICLE_COUNT = 2_000
MASS:  np.float64 = 1.0
RHO_0: np.float64 = 1.0
INF_R: np.float64 = 2.0
VISC:  np.float64 = 0.5
K:     np.float64 = 10.0
DAMP:  np.float64 = 0.7
INF_R_2 = INF_R ** 2
INF_R_6 = INF_R ** 6
INF_R_9 = INF_R ** 9
W_CONST = 315.0 / (64.0 * np.pi * INF_R_9)
GRAD_W_CONST = -45.0 / (np.pi * INF_R_6)
LAP_W_CONST = 45.0 / (np.pi * INF_R_6)
MAX_NEIGHBOURS = 32
NEIGHBOURING_VOXELS_COUNT = 27
VOXEL_SIZE = [INF_R, INF_R, INF_R]
BOX_SPACE_SIZE = [20 * INF_R, 20 * INF_R, 20 * INF_R]
PIPE_SPACE_SIZE = [20 * INF_R, 3 * INF_R, 3 * INF_R]
GRAVITY = [0.0, -2.0, 0.0]
HORIZONTAL_FORCE = [2.0, 0.0, 0.0]
PROJECT_DIRNAME = os.getcwd()
PARAMS_FILENAME = 'params.json'
OUT_DIRNAME = 'out'

__pipe: Pipe = PipeBuilder().with_starting_radius(3) \
        .add_roller_segment(1) \
        .add_increasing_segment(1, 1.2) \
        .add_roller_segment(1) \
        .add_lessening_segment(1, 1.2) \
        .add_roller_segment(1) \
        .transform(PIPE_SPACE_SIZE[0], PIPE_SPACE_SIZE[1]) \
        .get_result()


def inject_start_state():
    if SIM_MODE == SimMode.PIPE:
        return __start_state_pipe(__pipe)
    else:
        return __start_state_box_wall()


def inject_params():
    if SIM_MODE == SimMode.PIPE:
        return SimulationParameters(
            particle_count=PARTICLE_COUNT,
            external_force=HORIZONTAL_FORCE,
            duration=DURATION,
            fps=FPS,
            pipe=__pipe,
            space_size=BOX_SPACE_SIZE,
            voxel_size=VOXEL_SIZE
        )
    else:
        return SimulationParameters(
            particle_count=PARTICLE_COUNT,
            external_force=GRAVITY,
            duration=DURATION,
            fps=FPS,
            pipe=__pipe,
            space_size=BOX_SPACE_SIZE,
            voxel_size=VOXEL_SIZE
        )


def __start_state_box_wall() -> SimulationState:
    position = np.random \
        .random(PARTICLE_COUNT * 3) \
        .reshape((PARTICLE_COUNT, 3)) \
        .astype("float64")
    for i in range(PARTICLE_COUNT):
        position[i][0] *= BOX_SPACE_SIZE[0] * 0.1
        position[i][1] *= BOX_SPACE_SIZE[1]
        position[i][2] *= BOX_SPACE_SIZE[2]
    base_velocity = [1.5, -2.0, 0.0]
    offset_range = 1.0
    velocity = np.zeros(PARTICLE_COUNT * 3) \
        .reshape((PARTICLE_COUNT, 3)) \
        .astype("float64")
    for i in range(PARTICLE_COUNT):
        for dim in range(3):
            velocity[i][dim] = base_velocity[dim] + ((random() - 0.5) * offset_range)
    density = np.zeros(PARTICLE_COUNT).astype("float64")
    return SimulationState(position, velocity, density)


def __start_state_pipe(pipe: Pipe) -> SimulationState:
    position = np.zeros((PARTICLE_COUNT, 3), dtype=np.float64)
    for i in range(PARTICLE_COUNT):
        alpha = random() * 2 * np.pi
        radius = random() * pipe.segments[0].start_radius
        position[i][0] = 0
        position[i][1] = math.cos(alpha) * radius
        position[i][2] = math.sin(alpha) * radius
    velocity = np.zeros(PARTICLE_COUNT * 3) \
        .reshape((PARTICLE_COUNT, 3)) \
        .astype("float64")
    density = np.zeros(PARTICLE_COUNT).astype("float64")
    return SimulationState(position, velocity, density)



