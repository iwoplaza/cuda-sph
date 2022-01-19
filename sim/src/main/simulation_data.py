from __future__ import annotations
from dataclasses import dataclass
from numpy import ndarray
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
    external_force: ndarray # single 3d vector 
    simulation_duration: int # in seconds
    fps: int 
    pipe: Pipe
    space_dims: Tuple[int, int, int] # (width, height, depth)
    voxel_dim: Tuple[float, float, float] # (width, height, depth)


@dataclass
class SimulationState:
    position: ndarray # (n x 3)
    velocity: ndarray # (n x 3)
    density: ndarray  # (n)
    voxel: ndarray    # (n) idx = x + y*w + z*w*d