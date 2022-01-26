from __future__ import annotations
from utils import SimulationState, SimulationParameters
from state_generator import StateGenerator
from common.serializer.saver import Saver


def main():
    simulation_state = SimulationState()
    simulation_parameters = SimulationParameters()
    saver = Saver("simulation_data", simulation_parameters)
    for frame in StateGenerator(simulation_state, simulation_parameters):
        saver.save_next_epoch(frame)


if __name__ == '__main__':
    main()
