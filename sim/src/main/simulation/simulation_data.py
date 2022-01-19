from __future__ import annotations
from dataclasses import dataclass
from numpy import ndarray, asarray, float64, random, zeros
import numpy as np
from typing import Tuple, List


@dataclass
class Segment:
    end_point: Tuple[float, float, float]
    radius: float 
    prev_segment: Segment


@dataclass 
class Pipe:
    segments: List[Segment]


@dataclass
class SimulationParameters:
    n_particles: int
    external_force: ndarray  # (x,y,z)
    simulation_duration: int # in seconds
    fps: int 
    pipe: Pipe
    space_dims: ndarray # (x,y,z)
    voxel_dim: ndarray  # (x,y,z)


@dataclass
class SimulationState:
    position: ndarray # (n x 3)
    velocity: ndarray # (n x 3)
    density: ndarray  # (n)
    voxel: ndarray    # (n) idx = x + y*w + z*w*d


def get_default_start_sim_state(N) -> SimulationState:
    return SimulationState(
        position=np.random.random(N * 3).reshape((N, 3)).astype("float64"),
        velocity=np.random.random(N * 3).reshape((N, 3)).astype("float64"),
        density=zeros(N).astype("float64"),
        voxel=zeros(N).astype("int32")
    )

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
    
