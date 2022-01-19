from __future__ import annotations
from math import ceil
from numpy import float64, int32, zeros
from kernels import *
from simulation_data import *
from numba import cuda
from physics import *



class StateGenerator:
    def __init__(self, start_state: SimulationState, params: SimulationParameters) -> None:
        self.current_state = start_state
        self.current_frame_idx = -1
        self.params = params 
        self.dt = 1 / params.fps
        self.n_frames = params.simulation_duration / self.dt

    def __next__(self) -> SimulationState:
        self.current_frame_idx += 1

        if self.current_frame_idx == 0:
            return self.current_state
            
        if self.current_frame_idx >= self.n_frames:
            raise StopIteration

        self.current_state = self.__compute_next_state()

        return self.current_state

    def __iter__(self) -> StateGenerator:
        return self
    
    def __compute_next_state(self) -> SimulationState:
        N: int = self.params.n_particles

        # send data to gpu
        # a) data to be updated or populated
        d_position = cuda.to_device(self.current_state.position)
        d_velocity = cuda.to_device(self.current_state.velocity)
        d_new_density = cuda.to_device(zeros(N, dtype=float64))

        # b) arrays to be used during computations
        d_new_pressure_term = cuda.to_device(zeros((N, 3), dtype=float64))
        d_new_viscosity_term = cuda.to_device(zeros((N, 3), dtype=float64))
        d_external_force = cuda.to_device(self.params.external_force)

        # compute block size and grid size
        # TODO: compute it using gpu detection for optimization
        threads_per_grid: int = 64
        grids_per_block: int = ceil(self.params.n_particles / threads_per_grid)

        # run kernels
        density_kernel[grids_per_block, threads_per_grid](
            d_new_density, d_position, MASS, INF_R
        )
        cuda.synchronize()

        pressure_kernel[grids_per_block, threads_per_grid](
            d_new_pressure_term, d_new_density, d_position,
            MASS, INF_R, K, RHO_0
        )
        cuda.synchronize()

        viscosity_kernel[grids_per_block, threads_per_grid](
            d_new_viscosity_term, d_new_density, d_position, d_velocity,
            MASS, INF_R, VISC
        )
        cuda.synchronize()

        integrating_kernel[grids_per_block, threads_per_grid](
            d_position, d_velocity, d_external_force, d_new_pressure_term,
            d_new_viscosity_term, self.dt, MASS
        )
        cuda.synchronize()

        return SimulationState(
            d_position.copy_to_host(),
            d_velocity.copy_to_host(),
            d_new_density.copy_to_host(),
            zeros(N, dtype=int32)
        )


    