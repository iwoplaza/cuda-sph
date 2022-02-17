from __future__ import annotations
from sim.src.sph import NaiveSPHStrategy, VoxelSPHStrategy, AbstractSPHStrategy
from common.data_classes import SimulationState, SimulationParameters


class StateGenerator:
    def __init__(self,
                 start_state: SimulationState,
                 params: SimulationParameters) -> None:
        self.current_state = start_state
        self.current_frame_idx = 0
        self.n_frames = params.duration * params.fps
        self.sph_strategy: AbstractSPHStrategy = NaiveSPHStrategy(params)

    def __next__(self) -> SimulationState:
        if self.current_frame_idx >= self.n_frames:
            raise StopIteration

        print(f"Computing frame {self.current_frame_idx + 1}...")
        if self.current_frame_idx > 0:
            self.current_state = self.sph_strategy.compute_next_state(self.current_state)

        self.current_frame_idx += 1
        return self.current_state

    def __iter__(self) -> StateGenerator:
        return self
