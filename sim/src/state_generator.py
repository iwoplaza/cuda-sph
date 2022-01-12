from __future__ import annotations
from kernels.kernels import density_kernel
from state_generator import StateGenerator
from utils import *
from copy import deepcopy
import cupy as cp 



class StateGenerator:
    def __init__(self, start_state: SimulationState, params: SimulationParameters) -> None:
        self.current_state = start_state
        self.frame_idx = 0
        self.params = params 
        self.dt = 1 / params.fps
        self.n_frames = params.simulation_duration / self.dt

    
    def __cuda_pipeline(self) -> None:
        d_density = cp.array(self.current_state.density)
        d_position = cp.array(self.current_state.position)
        d_pressure = cp.array(self.current_state.pressure)
        d_velocity = cp.array(self.current_state.velocity)
        density_kernel[n_blocks, n_threads](d_density, d_position, )


    def __next__(self) -> SimulationState:
        if self.frame_idx >= self.n_frames:
            raise StopIteration
        self.frame_idx += 1
        return f"Frame number {self.frame_idx}"


    def __iter__(self) -> StateGenerator:
        return self