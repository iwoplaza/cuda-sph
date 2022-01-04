import numpy as np
import json
import os

import constants


class Saver:
    """
    Class for saving simulation data
    """

    def __init__(self, folder_path: str, pipe, particles_number) -> None:
        """
        :param folder_path: path to folder where place data simulation
        :param pipe: Pipe object
        :param particles_number: number of simulated particles
        """
        os.mkdir(folder_path)
        self.folder_path = folder_path
        json_object = {constants.PIPE: pipe, constants.PARTICLES_NUMBER: particles_number}
        with open(folder_path + constants.SETTINGS_FILE) as file:
            json.dump(json_object, file)
        self.current_epoch = 0

    def save_next_epoch(self, particles_state: np.ndarray, pressure_state: np.ndarray) -> None:
        """
        Saves epoch_state and increment internal epoch counter - use only once per epoch
        :param particles_state: Numpy matrix with information about positions and velocity of particles
        :param pressure_state: Numpy matrix with information about pressure
        """
        np.save(self.folder_path + constants.PARTICLES_FILE % self.current_epoch, particles_state)
        np.save(self.folder_path + constants.PRESSURE_FILE % self.current_epoch, pressure_state)
        self.current_epoch = self.current_epoch+1
