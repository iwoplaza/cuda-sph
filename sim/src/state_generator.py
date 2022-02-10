from __future__ import annotations

from common.data_classes import SimulationState, SimulationParameters
import sim.src.sph as sph


class StateGenerator:
    def __init__(self, start_state: SimulationState, params: SimulationParameters) -> None:
        self.current_state = start_state
        self.current_frame_idx = -1
        self.n_frames = params.simulation_duration * params.fps
        self.sph_strategy: sph.AbstractSPHStrategy = sph.VoxelSPHStrategy(params)

    def __next__(self) -> SimulationState:
        self.current_frame_idx += 1

        if self.current_frame_idx == 0:
            return self.current_state
            
        if self.current_frame_idx >= self.n_frames:
            raise StopIteration

        print(f"Computing {self.current_frame_idx}th frame...")
        self.current_state = self.sph_strategy.compute_next_state(self.current_state)

        return self.current_state

    def __iter__(self) -> StateGenerator:
        return self