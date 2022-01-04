import numpy as np
import json
import constants


class Loader:
    """
    Class responsible for loading saved simulation frames
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param folder_path: path to folder with data about simulation
        """
        self.folder_path = folder_path
        with open(folder_path + constants.SETTINGS_FILE) as file:
            self.json_object = json.load(file)

    def load_pipe(self):
        """
        :return: Pipe object using at simulation
        """
        return self.json_object[constants.PIPE]

    def load_particles_state(self, epoch: int) -> np.ndarray:
        """
        Load particle state in selected epoch
        :param epoch: Number of epoch to load data from
        :return: Numpy matrix with information about positions and velocity of particles
        """
        return np.load(self.folder_path + constants.PARTICLES_FILE % epoch)

    def load_pressure_data(self, epoch: int) -> np.ndarray:
        """
        Load pressure state in selected epoch
        :param epoch: Number of epoch to load pressure data
        :return: Numpy matrix with information about pressure
        """
        return np.load(self.folder_path + constants.PRESSURE_FILE % epoch)

    def get_particles_number(self) -> int:
        """
        :return: Number of simulated particles
        """
        return self.json_object[constants.PARTICLES_NUMBER]
