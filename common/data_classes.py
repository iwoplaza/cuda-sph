from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, List, Optional
import config
import logging 

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start_point: Tuple[np.float64, np.float64, np.float64] = (0.0, 0.0, 0.0)
    start_radius: np.float64 = 1.0
    end_radius: np.float64 = 1.0
    length: np.float64 = 1.0

    def to_numpy(self) -> np.ndarray:
        segment_values = list(self.start_point)
        segment_values.extend([self.start_radius, self.length])
        return np.array(segment_values, dtype=np.float64)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SimulationParameters:
    n_particles: np.int32 = config.DEFAULT_N_PARTICLES
    external_force: np.ndarray = field(default_factory=lambda: config.DEFAULT_EXT_FORCE)
    simulation_duration: np.int32 = config.DEFAULT_DURATION  # in seconds
    fps: np.int32 = config.DEFAULT_FPS
    pipe: Pipe = field(default_factory=lambda: Pipe(segments=[Segment()]), repr=False)
    space_size: np.ndarray = field(default_factory=lambda: config.DEFAULT_SPACE_SIZE)
    voxel_size: np.ndarray = field(default_factory=lambda: config.DEFAULT_VOXEL_SIZE)

    def __post_init__(self):
        # make numpy arrays read-only 
        self.external_force.setflags(write=False)
        self.space_size.setflags(write=False)
        self.voxel_size.setflags(write=False)
        logger.info(f"SimulationParameters object has been initialized to: {self}")


@dataclass(frozen=True)
class SimulationState:
    position: Optional[np.ndarray] = None  # (n x 3)
    velocity: Optional[np.ndarray] = None  # (n x 3)
    density:  Optional[np.ndarray] = None  # (n)


