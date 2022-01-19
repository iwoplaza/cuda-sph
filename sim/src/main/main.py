from __future__ import annotations

from serializer import Serializer
from simulation.simulation_data import get_default_sim_parameters, get_default_start_sim_state
from simulation.state_generator import StateGenerator


if __name__ == '__main__':

    params = get_default_sim_parameters()
    start_state = get_default_start_sim_state(params.n_particles)

    serializer = Serializer("results.txt")
    generator = StateGenerator(start_state, params)

    for state in generator:
        serializer.serialize(state)
    serializer.close()
