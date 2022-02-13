from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List
from sim.src.constants import INF_R

DEFAULT_N_PARTICLES = 100_000
DEFAULT_SPACE_SIDE_LENGTH = 20 * INF_R
DEFAULT_VOXEL_SIDE_LENGTH = INF_R


@dataclass
class Segment:
    start_point: Tuple[np.float64, np.float64, np.float64] = (0.0, 0.0, 0.0)
    start_radius: np.float64 = 1.0
    end_radius: np.float64 = 1.0
    length: np.float64 = 1.0

    def to_numpy(self) -> np.ndarray:
        segment_values = list(self.start_point)
        segment_values.extend([self.start_radius, self.length])
        return np.array(segment_values, dtype=np.float64)


@dataclass
class Pipe:
    segments: List[Segment]

    def to_numpy(self) -> np.ndarray:
        to_stack = []
        for segment in self.segments:
            to_stack.append(segment.to_numpy())
        last = self.segments[-1].to_numpy()
        last[0] = last[0] + last[-1]
        last[3] = self.segments[-1].end_radius
        to_stack.append(last)
        return np.stack(to_stack)

    def transform(self, space_size: float, max_radius: float = None) -> None:
        """
        Transform pipe to fit into cube.
        - rescales lengths such that pipe goes from 0 to space_size in x axis
        - y and z axis has values: space_size/2
        - rays are rescaled that max has specified values

        :param space_size: Value of cube edge
        :param max_radius: Radius of the maximum radius after rescaling if None no rescale is done
        """
        total_length = 0.
        for segment in self.segments:
            total_length += segment.length
        length_scale = space_size/total_length
        current_x = 0.
        for segment in self.segments:
            segment.start_point = (current_x, space_size/2, space_size/2)
            segment.length = segment.length*length_scale
            current_x += segment.length

        if max_radius:
            current_max_radius = 0.
            for segment in self.segments:
                if segment.start_radius > current_max_radius:
                    current_max_radius = segment.start_radius
                if segment.end_radius > current_max_radius:
                    current_max_radius = segment.end_radius
            scale = max_radius / current_max_radius
            print(scale)
            for segment in self.segments:
                segment.start_radius = segment.start_radius * scale
                segment.end_radius = segment.end_radius * scale


@dataclass
class SimulationParameters:
    n_particles: np.int32 = DEFAULT_N_PARTICLES
    external_force: np.ndarray = np.array([50.5, 0, 0], dtype=np.float64)  # (x,y,z)
    simulation_duration: np.int32 = 1  # in seconds
    fps: np.int32 = 6
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
