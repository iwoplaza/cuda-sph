from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List
from sim.src.main.physics.constants import INF_R

DEFAULT_N_PARTICLES = 50
DEFAULT_SPACE_SIDE_LENGTH = 7
DEFAULT_VOXEL_SIDE_LENGTH = INF_R * 2


@dataclass
class Segment:
    start_point: Tuple[np.float64, np.float64, np.float64] = (0, 0, 0)
    start_radius: np.float64 = 1
    end_radius: np.float64 = 1
    length: np.float64 = 1


@dataclass
class Pipe:
    segments: List[Segment]


@dataclass
class SimulationParameters:
    n_particles: np.int32 = DEFAULT_N_PARTICLES
    external_force: np.ndarray = np.array([0.5, 0, 0], dtype=np.float64)  # (x,y,z)
    simulation_duration: np.int32 = 2  # in seconds
    fps: np.int32 = 2
    pipe: Pipe = Pipe(segments=[Segment()])
    space_size: np.ndarray = np.array([DEFAULT_SPACE_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)
    voxel_size: np.ndarray = np.array([DEFAULT_VOXEL_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)


@dataclass
class SimulationState:
    position: np.ndarray = np.array(1)  # (n x 3)
    velocity: np.ndarray = np.array(1)  # (n x 3)
    density: np.ndarray = np.array(1)   # (n)

    def set_random_from_params(self, params):
        # shuffle particles inside inside whole space (for fun)
        self.position = np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64")
        for i in range(params.n_particles):
            for dim in range(3):
                self.position[i][dim] *= params.space_size[dim]
        self.velocity = np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64")
        self.density = np.zeros(params.n_particles).astype("float64")
