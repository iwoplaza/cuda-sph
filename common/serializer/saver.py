import numpy as np
import json
import os
from common.data_classes import SimulationState, SimulationParameters


SETTINGS_FILE = "/settings.json"


class Saver:
    """
    Class for saving simulation data.
    The saver saves file in specified directory and save:
    - settings.json for parameters of simulation
    - name.npy for parameters of simulation that are numpy arrays
    - name_%d.npy for simulation state for specified name
    """

    def __init__(self, folder_path: str, simulation_parameters: SimulationParameters) -> None:
        """
        :param folder_path: path to folder where place data simulation
        :param simulation_parameters: Parameters of simulation
        """
        self.__folder_path = folder_path
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self._json_object = {}
        self.__save_parameters(simulation_parameters)

        self.__current_epoch = 0

    def __save_parameters(self, simulation_parameters: SimulationParameters) -> None:
        # print("Params: ", simulation_parameters)
        for name, value in vars(simulation_parameters).items():
            self.__manage_element(name, value)

        # print(self._json_object)
        with open(self.__folder_path + SETTINGS_FILE, "w") as file:
            json.dump(self._json_object, file, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __manage_element(self, name: str, value) -> None:
        """
        Adds element to be saved to json or saves in .npy if element is of type np.ndarray

        :param name: Name of SimulationParameters variable
        :param value: Value of SimulationParameters variable
        """
        if type(value) is np.ndarray:
            np.save(self.__folder_path + "/" + name + ".npy", value)
        else:
            # print(name, value)
            self._json_object[name] = value

    def save_next_state(self, simulation_state: SimulationState) -> None:
        """
        Saves epoch_state and increment internal epoch counter - use only once per epoch

        :param simulation_state: Dataclass with information about positions and velocity of particles
        Assuming that all are of type np.ndarray.
        """
        for name, value in vars(simulation_state).items():
            np.save(self.__folder_path + "/" + name + "_" + str(self.__current_epoch), value)
        self.__current_epoch = self.__current_epoch + 1


if __name__ == '__main__':
    params = SimulationParameters()
    Saver("data", params)
