from __future__ import annotations

from common.main.data_classes.simulation_data_classes import get_default_sim_parameters, get_default_start_sim_state
from common.main.serializer.saver import Saver
from simulation.state_generator import StateGenerator

if __name__ == '__main__':

    params = get_default_sim_parameters()
    start_state = get_default_start_sim_state(params.n_particles)

    serializer = Saver("simulation_out", params)
    generator = StateGenerator(start_state, params)

    for state in generator:
        serializer.save_next_state(state)
