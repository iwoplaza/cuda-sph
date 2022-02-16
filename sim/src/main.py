from __future__ import annotations
import numpy as np
from common import start_states
from common.pipe_builder import PipeBuilder
from config import LONG_SPACE_SIZE
from state_generator import StateGenerator
from common.data_classes import SimulationParameters
from common.serializer.saver import Saver

if __name__ == '__main__':

    pipe = PipeBuilder().with_ending_radius(3) \
        .transform(LONG_SPACE_SIZE[0], LONG_SPACE_SIZE[1]) \
        .get_result()
    params = SimulationParameters(
        simulation_duration=3,
        space_size=LONG_SPACE_SIZE,
        pipe=pipe
    )
    #start_state = start_states.pouring(params)
    start_state = start_states.inside_pipe(params, pipe)

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
        # print(f"pos: {np.round(state.position[0], 1).tolist()},"
        #       f" vel: {np.round(state.velocity[0], 5).tolist()},"
        #       f" force:{np.round(generator.sph_strategy.result_force[0], 1).tolist()}")

    print("Simulation finished.")
