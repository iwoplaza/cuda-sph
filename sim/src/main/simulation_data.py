from __future__ import annotations
from dataclasses import dataclass
from numpy import int32, ndarray, asarray, float64, zeros
import numpy as np
from typing import Tuple, List
from physics.constants import INF_R


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
    voxel_size: ndarray  # (width, height, depth)
    voxel_count: ndarray # (x, y, z)
    space_size: ndarray  # (width, height, depth)


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
    voxel_size = asarray([INF_R * 2, INF_R * 2, INF_R * 2], dtype=float64)
    voxel_count = asarray([1000, 1000, 1000], dtype=int32)
    space_width = voxel_size[0] * voxel_count[0]
    space_height = voxel_size[1] * voxel_count[1]
    space_depth = voxel_size[2] * voxel_count[2]
    space_size = asarray([space_width, space_height, space_depth], dtype=float64)

    return SimulationParameters(
        n_particles=1000,
        external_force=asarray([0.5, 0.0, 0.0], dtype=float64),
        simulation_duration=1,
        fps=30,
        pipe=Pipe(segments=None),
        voxel_size=voxel_size,
        voxel_count=voxel_count,
        space_size=space_size
    ) 
    
