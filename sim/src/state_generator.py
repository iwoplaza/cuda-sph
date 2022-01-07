from __future__ import annotations
from state_generator import StateGenerator
from utils import *
from copy import deepcopy


class StateGenerator:
    def __init__(self, start_state: SimulationState, params: SimulationParameters) -> None:
        self.current_state = start_state
        self.frame_idx = 0
        self.params = params 
        self.dt = 1 / params.fps
        self.n_frames = params.simulation_duration / self.dt

    
    def __cuda_pipeline(self) -> None:
        pass


    def __next__(self) -> SimulationState:
        if self.frame_idx >= self.n_frames:
            raise StopIteration
        self.frame_idx += 1
        return f"Frame number {self.frame_idx}"

    def __iter__(self) -> StateGenerator:
        return self