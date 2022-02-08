from __future__ import annotations

from dataclasses import dataclass
from numpy import ndarray, asarray, float64, random, zeros
import numpy as np
from typing import Tuple, List


@dataclass
class Segment:
    start_point: Tuple[float, float, float] = (0, 0, 0)
    start_radius: float = 1
    end_radius: float = 1
    length: float = 1


@dataclass
class Pipe:
    segments: List[Segment]


@dataclass
class SimulationParameters:
    n_particles: int = 10000
    external_force: ndarray = np.array([2, 4, 5])  # (x,y,z)
    simulation_duration: int = 60  # in seconds
    fps: int = 30
    pipe: Pipe = Pipe(segments=[Segment()])
    space_dims: ndarray = np.array([3, 2, 1])  # (x,y,z)
    voxel_dim: ndarray = np.array([1, 2, 3])  # (x,y,z)


@dataclass
class SimulationState:
    position: ndarray  # (n x 3)
    velocity: ndarray  # (n x 3)
    density: ndarray  # (n)
    voxel: ndarray  # (n) idx = x + y*w + z*w*d


def get_default_start_sim_state(N) -> SimulationState:
    return SimulationState(
        position=np.random.random(N * 3).reshape((N, 3)).astype("float64"),
        velocity=np.random.random(N * 3).reshape((N, 3)).astype("float64"),
        density=zeros(N).astype("float64"),
        voxel=zeros(N).astype("int32")
    )

# TODO this is to remove or refactor
def get_default_sim_parameters() -> SimulationParameters:
    return SimulationParameters(
        n_particles=100,
        external_force=asarray([0.5, 0.0, 0.0], dtype=float64),
        simulation_duration=1,
        fps=30,
        pipe=Pipe(segments=None),
        space_dims=(1, 1, 1),
        voxel_dim=(1e-4, 1e-4, 1e-4)
    )
