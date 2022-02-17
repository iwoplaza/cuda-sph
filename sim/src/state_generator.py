from __future__ import annotations
import logging
import numpy as np
from timeit import default_timer as timer

import config
from sim.src.sph import NaiveSPHStrategy, VoxelSPHStrategy, AbstractSPHStrategy
from common.data_classes import SimulationState, SimulationParameters
import sim.src.sph as sph

logger = logging.getLogger(__name__)


class StateGenerator:
    def __init__(self,
                 start_state: SimulationState,
                 params: SimulationParameters) -> None:
        self.current_state = start_state
        self.current_frame_idx = 0
        self.n_frames = params.duration * params.fps
        self.sph_strategy: AbstractSPHStrategy = (NaiveSPHStrategy(params)
                                                  if config.SIM_STRATEGY == 'NAIVE'
                                                  else VoxelSPHStrategy(params))
        self.__init_log(params)

    def __next__(self) -> SimulationState:
        if self.current_frame_idx >= self.n_frames:
            raise StopIteration

        self.__computation_start_log()

        self.current_state = self.sph_strategy.compute_next_state(self.current_state)

        self.__computation_end_log()

        self.current_frame_idx += 1
        return self.current_state

    def __iter__(self) -> StateGenerator:
        return self

    def __computation_start_log(self):
        logger.info(f"Computing frame {self.current_frame_idx + 1}...")
        self.__start = timer()

    def __computation_end_log(self):
        logger.info(
            f"frame {self.current_frame_idx } / {self.n_frames}"
            f" computed in {timer() - self.__start:.4f} seconds"
        )

    def __init_log(self, params) -> None:
        logger.info(f"Simulation parameters: {params}")
        logger.info(f"Pipe used: {params.pipe}")
        logger.info(f"Algorithm used: {self.sph_strategy.__class__}")
