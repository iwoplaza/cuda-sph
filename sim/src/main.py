from __future__ import annotations
import numpy as np
import config
from common.serializer.saver import Saver
from state_generator import StateGenerator

if __name__ == '__main__':

    saver = Saver(config.OUT_DIRNAME, config.params)
    generator = StateGenerator(config.start_state, config.params)

    print(f"Running simulation with {config.params.particle_count} particles.")
    print(f"Space size: {np.round(config.params.space_size, 2)}")
    print(f"Algorithm used: {config.SIM_STRATEGY}")
    print(f"Duration time: {config.params.duration} seconds. Framerate: {config.params.fps}. "
          f"Total frames number: {config.params.fps * config.params.duration}.")
    print(f"Thread layout: grid size {generator.sph_strategy.grid_size}, "
          f"block size {generator.sph_strategy.block_size}")

    for state in generator:
        saver.save_next_state(state)

    print("Simulation finished.")
