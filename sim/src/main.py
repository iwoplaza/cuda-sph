from __future__ import annotations
import numpy as np
from common import start_states
from sim.src.state_generator import StateGenerator
from common.data_classes import SimulationParameters
from common.serializer.saver import Saver

if __name__ == '__main__':

    params = SimulationParameters()
    start_state = start_states.pouring(params)

    saver = Saver("simulation_out", params)
    generator = StateGenerator(start_state, params)

    for state in generator:
        # saver.save_next_state(state)
        print(f"pos: {np.round(state.position[0], 1).tolist()},"
              f" vel: {np.round(state.velocity[0], 5).tolist()},"
              f" force:{np.round(generator.sph_strategy.result_force[0], 1).tolist()}")
    print("Simulation finished.")
