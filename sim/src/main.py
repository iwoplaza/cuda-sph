from __future__ import annotations
import numpy as np
import config
from common.serializer.saver import Saver
from state_generator import StateGenerator

if __name__ == '__main__':

    params = config.inject_params()
    start_state = config.inject_start_state()
    saver = Saver(config.OUT_DIRNAME, params)
    generator = StateGenerator(start_state, params)

    print(f"Running simulation with {params.particle_count} particles.")
    print(f"Space size: {np.round(params.space_size, 2)}")
    print(f"Algorithm used: {generator.sph_strategy.__class__.__name__}")
    print(f"Duration time: {params.duration} seconds. Framerate: {params.fps}. "
          f"Total frames number: {params.fps * params.duration}.")
    print(f"Thread layout: grid size {generator.sph_strategy.grid_size}, "
          f"block size {generator.sph_strategy.block_size}")

    for state in generator:
        saver.save_next_state(state)

    print("Simulation finished.")
