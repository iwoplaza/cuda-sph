from __future__ import annotations
from dataclasses import dataclass
from numpy import ndarray, asarray, float64, random, zeros, int32
import numpy as np
from typing import Tuple, List
from sim.src.main.physics.constants import INF_R


DEFAULT_N_PARTICLES = 10
DEFAULT_SPACE_SIZE = 100
DEFAULT_VOXEL_SIZE = INF_R * 2


@dataclass
class Segment:
    start_point: Tuple[float64, float64, float64] = (0, 0, 0)
    start_radius: float64 = 1
    end_radius: float64 = 1
    length: float64 = 1


@dataclass
class Pipe:
    segments: List[Segment]


@dataclass
class SimulationParameters:
    n_particles: int32 = DEFAULT_N_PARTICLES
    external_force: ndarray = np.array([0.5, 0, 0], dtype=float64)  # (x,y,z)
    simulation_duration: int32 = 60  # in seconds
    fps: int32 = 30
    pipe: Pipe = Pipe(segments=[Segment()])
    space_size: ndarray = np.array([DEFAULT_SPACE_SIZE for _ in range(3)], dtype=float64)  # (x,y,z)
    voxel_size: ndarray = np.array([DEFAULT_VOXEL_SIZE for _ in range(3)], dtype=float64)  # (x,y,z)


@dataclass
class SimulationState:
    position: ndarray  # (n x 3)
    velocity: ndarray  # (n x 3)
    density: ndarray  # (n)
    voxel: ndarray  # (n) idx = x + y*w + z*w*d

    def __init__(self, N, space_size):
        # shuffle particles inside inside whole space (for fun)
        position = np.random.random(N * 3).reshape((N, 3)).astype("float64")
        for i in range(len(position)):
            for dim in range(3):
                position[i][dim] *= space_size[dim]
        self.position = position

        self.velocity = np.random.random(N * 3).reshape((N, 3)).astype("float64")
        self.density = zeros(N).astype("float64")
        self.voxel = zeros(N).astype("int32")


