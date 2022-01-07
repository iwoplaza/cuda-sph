from __future__ import annotations
from dataclasses import dataclass
from numpy.typing import NDArray
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
    partices_radius: float
    pipe: Pipe
    influence_radius: float
    space_dims: Tuple[int, int, int] # (width, height, depth)
    voxel_dim: Tuple[int, int, int] # (width, height, depth)


@dataclass
class SimulationState:
    position: NDArray # 3d
    velocity: NDArray # 3d
    density: NDArray
    pressure: NDArray
    voxels: NDArray # 1d i = x + y*w + z*w*d ~ [x, y, z]