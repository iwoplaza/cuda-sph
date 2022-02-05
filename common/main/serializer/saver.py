import numpy as np
import json
import os

import common.main.serializer.constants as constants
from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters


def prepare_json(simulation_parameters: SimulationParameters):
    json_object = {constants.SIMULATION_DURATION: simulation_parameters.simulation_duration,
                   constants.FPS: simulation_parameters.fps,
                   constants.PARTICLES_NUMBER: simulation_parameters.n_particles,
                   constants.SPACE_DIMS: simulation_parameters.space_dims,
                   constants.VOXEL_DIM: simulation_parameters.voxel_dim,
                   constants.PIPE: None}
    return json_object


class Saver:
    """
    Class for saving simulation data.
    The saver saves file in specified directory and save:
    - settings.json for parameters of simulation
    - particles_%d.npy: to save state about particles positions
    - velocity_%d.npy: to save state about particles velocity
    - pressure_%d.npy: to save state about pressure
    """

    def __init__(self, folder_path: str, simulation_parameters: SimulationParameters) -> None:
        """
        :param folder_path: path to folder where place data simulation
        :param simulation_parameters: Parameters of simulation
        """
        self.__folder_path = folder_path
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        json_object = prepare_json(simulation_parameters)
        with open(folder_path + constants.SETTINGS_FILE, "w") as file:
            json.dump(json_object, file)

        self.__current_epoch = 0

    def save_next_state(self, simulation_state: SimulationState) -> None:
        """
        Saves epoch_state and increment internal epoch counter - use only once per epoch

        :param simulation_state: Numpy matrix with information about positions and velocity of particles
        """
        np.save(self.__folder_path + constants.POSITION_FILE % self.__current_epoch, simulation_state.position)
        np.save(self.__folder_path + constants.VELOCITY_FILE % self.__current_epoch, simulation_state.velocity)
        np.save(self.__folder_path + constants.PRESSURE_FILE % self.__current_epoch, simulation_state.density)
        self.__current_epoch = self.__current_epoch + 1
