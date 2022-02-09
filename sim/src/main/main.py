from __future__ import annotations
from state_generator import StateGenerator
from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters
from common.main.serializer.saver import Saver

if __name__ == '__main__':

    params = SimulationParameters()
    start_state = SimulationState()
    start_state.set_random_from_params(params)

    saver = Saver("simulation_out", params)
    generator = StateGenerator(start_state, params)

    print(f"Running simulation with {params.n_particles} particles " 
          f"in ({params.space_size[0]}, {params.space_size[1]}, "
          f"{params.space_size[2]}) space")
    print(f"Duration time: {params.simulation_duration} seconds. "
          f"Framerate: {params.fps}. "
          f"Total frames number: {params.fps * params.simulation_duration}")

    for state in generator:
        saver.save_next_state(state)
