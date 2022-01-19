from __future__ import annotations
from simulation_data import get_default_sim_parameters, get_default_start_sim_state
from state_generator import StateGenerator

from serializer import Serializer



if __name__ == '__main__':

    params = get_default_sim_parameters()
    start_state = get_default_start_sim_state(params.n_particles)

    serializer = Serializer("results.txt")
    generator = StateGenerator(start_state, params)

    print(f"Running simulation with {params.n_particles} particles " 
          f"in ({params.space_size[0]}, {params.space_size[1]}, {params.space_size[2]}) space")

    for state in generator:
        serializer.serialize(state)
    serializer.close()
