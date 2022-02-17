"""
    This module has to be located in the project root directory.
    This module is really important, many modules depend on that.
"""
import math
import pathlib
import numpy as np
from random import random
from common.data_classes import Pipe, Segment, SimulationParameters, SimulationState
from common.pipe_builder import PipeBuilder


SIM_MODE = 'BOX'       # 'BOX' or 'PIPE'
DURATION = 15
FPS = 20
PARTICLE_COUNT = 50_000
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

ROOT_PROJ_DIRNAME = pathlib.Path(__file__).parent.resolve()  # absolute path to cuda-sph
ASSETS_DIRNAME = 'vis/assets'
PARAMS_FILENAME = 'params.json'
OUT_DIRNAME = 'out'


def __create_params_pipe():
    return SimulationParameters(
            particle_count=PARTICLE_COUNT,
            external_force=HORIZONTAL_FORCE,
            duration=DURATION,
            fps=FPS,
            pipe=pipe,
            space_size=PIPE_SPACE_SIZE,
            voxel_size=VOXEL_SIZE
        )


def __create_params_box():
    return SimulationParameters(
            particle_count=PARTICLE_COUNT,
            external_force=GRAVITY,
            duration=DURATION,
            fps=FPS,
            pipe=Pipe(),
            space_size=BOX_SPACE_SIZE,
            voxel_size=VOXEL_SIZE
        )


def __build_pipe():
    return PipeBuilder().with_starting_radius(1) \
        .add_roller_segment(1) \
        .add_increasing_segment(1, 1.2) \
        .add_roller_segment(1) \
        .add_lessening_segment(1, 1.2) \
        .add_roller_segment(1) \
        .transform(PIPE_SPACE_SIZE[0], PIPE_SPACE_SIZE[1]) \
        .get_result()


def __start_state_box_wall() -> SimulationState:
    position = np.random \
        .random(PARTICLE_COUNT * 3) \
        .reshape((PARTICLE_COUNT, 3)) \
        .astype("float64")
    for i in range(PARTICLE_COUNT):
        position[i][0] *= BOX_SPACE_SIZE[0] * 0.1
        position[i][1] *= BOX_SPACE_SIZE[1]
        position[i][2] *= BOX_SPACE_SIZE[2]
    base_velocity = [1.5, -5.0, -5.0]
    offset_range = 1.0
    velocity = np.zeros(PARTICLE_COUNT * 3) \
        .reshape((PARTICLE_COUNT, 3)) \
        .astype("float64")
    for i in range(PARTICLE_COUNT):
        for dim in range(3):
            velocity[i][dim] = base_velocity[dim] + ((random() - 0.5) * offset_range)
    density = np.zeros(PARTICLE_COUNT).astype("float64")
    return SimulationState(position, velocity, density)


def __start_state_inside_pipe() -> SimulationState:
    position = np.zeros((PARTICLE_COUNT, 3), dtype=np.float64)
    velocity = np.zeros((PARTICLE_COUNT, 3), dtype=np.float64)
    density = np.zeros(PARTICLE_COUNT, dtype=np.float64)
    # place particles randomly inside pipe with horizontal velocity
    for i in range(PARTICLE_COUNT):
        x = random() * pipe.get_length()
        segment_idx = 0
        offset = pipe.segments[segment_idx].start_point
        r_max = pipe.radius_at(x)
        r = math.sqrt(random()) * r_max * 0.98
        theta = random() * 2.0 * math.pi
        position[i, 0] = x
        position[i, 1] = offset[1] + r * math.cos(theta)
        position[i, 2] = offset[2] + r * math.sin(theta)
        velocity[i][0] = 0.0
    return SimulationState(
        position,
        velocity,
        density
    )


if SIM_MODE == 'BOX':
    pipe = Pipe([Segment()])
    params = __create_params_box()
    start_state = __start_state_box_wall()
elif SIM_MODE == 'PIPE':
    pipe = __build_pipe()
    params = __create_params_pipe()
    start_state = __start_state_inside_pipe()
else:
    raise Exception(f'Wrong simulation mode! ({SIM_MODE})')
SIM_STRATEGY = 'NAIVE'  # 'NAIVE' or 'VOXEL'



