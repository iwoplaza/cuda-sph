from __future__ import annotations
from dataclasses import dataclass
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
    particle_mass: float
    simulation_duration: int
    fps: int
    particles_number: int
    particles_radius: float
    pipe: Pipe
    influence_radius: float
    space_dims: Tuple[int, int, int]  # (width, height, depth)
    voxel_dim: Tuple[int, int, int]  # (width, height, depth)


@dataclass
class SimulationState:
    position: np.ndarray  # 3d
    velocity: np.ndarray  # 3d
    density: np.ndarray
    pressure: np.ndarray
    voxels: np.ndarray  # 1d i = x + y*w + z*w*d ~ [x, y, z]
