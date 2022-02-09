from __future__ import annotations
from dataclasses import dataclass
from numpy import ndarray, float64, random, zeros, int32, asarray
import numpy as np
from typing import Tuple, List
from sim.src.main.physics.constants import INF_R

DEFAULT_N_PARTICLES = 6
DEFAULT_SPACE_SIDE_LENGTH = 7
DEFAULT_VOXEL_SIDE_LENGTH = INF_R * 2


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
    simulation_duration: int32 = 2  # in seconds
    fps: int32 = 2
    pipe: Pipe = Pipe(segments=[Segment()])
    space_size: ndarray = np.array([DEFAULT_SPACE_SIDE_LENGTH for _ in range(3)], dtype=float64)  # (x,y,z)
    voxel_size: ndarray = np.array([DEFAULT_VOXEL_SIDE_LENGTH for _ in range(3)], dtype=float64)  # (x,y,z)


@dataclass
class SimulationState:
    position: ndarray = np.array(1)  # (n x 3)
    velocity: ndarray = np.array(1)  # (n x 3)
    density: ndarray = np.array(1)   # (n)

    def set_random_from_params(self, params):
        # shuffle particles inside inside whole space (for fun)
        self.position = np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64")
        for i in range(params.n_particles):
            for dim in range(3):
                self.position[i][dim] *= params.space_size[dim]
        self.velocity = np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64")
        self.density = zeros(params.n_particles).astype("float64")
