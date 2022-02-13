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

    # # TMP:
    # from timeit import timeit
    # space_size = generator.sph_strategy.params.space_size
    # finished = False
    # while not finished:
    #     state: SimulationState
    #     try:
    #         print(timeit(lambda: generator.__next__(), number=1))
    #         # look at one of the particles
    #         pos = generator.sph_strategy.old_state.position[0]
    #         print(f"1st particle position: {np.round(pos, 3)}")
    #     except StopIteration:
    #         finished = True

    print("Simulation finished.")
