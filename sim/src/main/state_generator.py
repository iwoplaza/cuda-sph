from __future__ import annotations

from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters
from physics.simulator import Simulator


class StateGenerator:
    def __init__(self, start_state: SimulationState, params: SimulationParameters) -> None:
        self.current_state = start_state
        self.current_frame_idx = -1
        self.n_frames = params.simulation_duration * params.fps
        self.simulator = Simulator(params)

    def __next__(self) -> SimulationState:
        self.current_frame_idx += 1

        if self.current_frame_idx == 0:
            return self.current_state
            
        if self.current_frame_idx >= self.n_frames:
            raise StopIteration

        self.current_state = self.simulator.compute_next_state(self.current_state)

        return self.current_state

    def __iter__(self) -> StateGenerator:
        return self