from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, List


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

    def radius_at(self, t: float):
        """
        Finds radius of a segment at given x.
        """
        if self.start_radius == self.end_radius:
            return self.start_radius
        radius_change = self.end_radius - self.start_radius
        x_in_seg = t - self.start_point[0]
        return self.start_radius + radius_change * (x_in_seg / self.length)


@dataclass(frozen=True)
class Pipe:
    segments: List[Segment] = field(default_factory=lambda: [])

    def to_numpy(self) -> np.ndarray:
        if len(self.segments) == 0:
            return np.asarray([])
        to_stack = []
        for segment in self.segments:
            to_stack.append(segment.to_numpy())
        last = self.segments[-1].to_numpy()
        last[0] = last[0] + last[-1]
        last[3] = self.segments[-1].end_radius
        to_stack.append(last)
        return np.stack(to_stack)

    def get_length(self):
        length = 0.0
        for segment in self.segments:
            length += segment.length
        return length

    def find_segment(self, t: float):
        """
        Finds the segment which contains given point at x dimension.
        """
        for i, segment in enumerate(self.segments):
            start = segment.start_point[0]
            end = segment.start_point[0] + segment.length
            if start <= t <= end:
                return i
        return -1

    def radius_at(self, t: float):
        """
        Finds radius of a pipe at given x.
        """
        return self.segments[self.find_segment(t)].radius_at(t)


@dataclass(frozen=True)
class SimulationParameters:
    particle_count: np.int32 = 100
    external_force: np.ndarray = np.asarray([0, 0, 0])
    duration:       np.int32 = 10
    fps:            np.int32 = 20
    pipe:           Pipe = Pipe([Segment()])
    space_size:     np.ndarray = np.asarray([1, 1, 1])
    voxel_size:     np.ndarray = np.asarray([1, 1, 1])


@dataclass(frozen=True)
class SimulationState:
    position: np.ndarray = np.asarray([[1, 1, 1]])  # (n x 3)
    velocity: np.ndarray = np.asarray([[1, 1, 1]])  # (n x 3)
    density:  np.ndarray = np.asarray([1])          # (n)



