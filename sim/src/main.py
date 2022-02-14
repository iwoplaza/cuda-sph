from __future__ import annotations
import numpy as np
from common import start_states
from state_generator import StateGenerator
from common.data_classes import SimulationParameters
from common.serializer.saver import Saver

if __name__ == '__main__':

    params = SimulationParameters()
    start_state = start_states.pouring(params)

    saver = Saver("simulation_out", params)
    generator = StateGenerator(start_state, params)

    print(f"Running simulation with {params.n_particles} particles.")
    print(f"Space size: {np.round(params.space_size, 2)}")
    print(f"Algorithm used: {generator.sph_strategy.__class__}")
    print(f"Duration time: {params.simulation_duration} seconds. "
          f"Framerate: {params.fps}. "
          f"Total frames number: {params.fps * params.simulation_duration}.")
    print(f"Thread layout: grid size {generator.sph_strategy.grid_size}, "
          f"block size {generator.sph_strategy.block_size}")

    for state in generator:
        saver.save_next_state(state)
        # print(f"\nComputing frame {generator.current_frame_idx + 1}...")
        # print(f"pos: {np.round(state.position[0], 1).tolist()}, "
        #       f"vel: {np.round(state.velocity[0], 1).tolist()}, "
        #       f"dens: {np.round(generator.sph_strategy.tmp_density[0], 1)}, "
        #       f"press: {np.round(generator.sph_strategy.tmp_pressure[0], 1)}, "
        #       f"visc: {np.round(generator.sph_strategy.tmp_visc_term[0], 1)}, "
        #       f"force:{np.round(generator.sph_strategy.result_force[0], 1).tolist()}")

    print("Simulation finished.")
