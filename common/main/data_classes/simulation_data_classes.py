from __future__ import annotations
from dataclasses import dataclass
from numpy import int32, ndarray, asarray, float64, zeros
import numpy as np
from typing import Tuple, List
from sim.src.main.physics.constants import INF_R

DEFAULT_SPACE_SIZE = [100, 100, 100]
DEFAULT_N_PARTICLES = 10
DEFAULT_VOXEL_SIZE = [INF_R * 2, INF_R * 2, INF_R * 2]


@dataclass
class Segment:
    end_point: List[float64]
    radius: float64
    prev_segment: Segment


@dataclass
class Pipe:
    segments: List[Segment]


@dataclass
class SimulationParameters:
    n_particles: int
    external_force: List  # (x,y,z)
    simulation_duration: int  # in seconds
    fps: int
    pipe: Pipe
    voxel_size: List
    space_size: List


@dataclass
class SimulationState:
    position: ndarray  # (n x 3)
    velocity: ndarray  # (n x 3)
    density: ndarray  # (n)
    voxel: ndarray  # (n) idx = x + y*w + z*w*d


def get_default_start_sim_state(N) -> SimulationState:
    position = np.random.random(N * 3).reshape((N, 3)).astype("float64")
    for i in range(len(position)):
        for dim in range(3):
            position[i][dim] *= DEFAULT_SPACE_SIZE[dim]
    return SimulationState(
        position=position,
        velocity=np.random.random(N * 3).reshape((N, 3)).astype("float64"),
        density=zeros(N).astype("float64"),
        voxel=zeros(N).astype("int32")
    )


def get_default_sim_parameters() -> SimulationParameters:
    return SimulationParameters(
        n_particles=10,
        external_force=[0.5, 0.0, 0.0],
        simulation_duration=1,
        fps=30,
        pipe=Pipe(segments=list()),
        voxel_size=DEFAULT_VOXEL_SIZE,
        space_size=DEFAULT_SPACE_SIZE
    )
