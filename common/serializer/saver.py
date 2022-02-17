import dataclasses
import numpy as np
import json
import os
import config
from common.data_classes import SimulationState, SimulationParameters


class Saver:
    """
    Class for saving simulation data.
    The saver saves file in specified directory and save:
    - settings.json for parameters of simulation
    - name_%d.npy for simulation state for specified name
    """

    def __init__(self, out_dirname: str, params: SimulationParameters) -> None:
        """
        :param out_dirname: relative (to root) path to the directory where place data simulation
        :param params: Parameters of simulation
        """
        self.__out_folder_path = os.path.join(config.ROOT_PROJ_DIRNAME, out_dirname)
        if not os.path.exists(self.__out_folder_path):
            os.mkdir(self.__out_folder_path)
        self.__save_parameters(params)
        self.__current_epoch = 0

    def save_next_state(self, state: SimulationState) -> None:
        """
        Saves epoch_state and increment internal epoch counter - use only once per epoch
        :param state: Dataclass with information about positions and velocity of particles
        Assuming that all are of type np.ndarray.
        """
        for name, value in vars(state).items():
            np.save(os.path.join(self.__out_folder_path, name + "_" + str(self.__current_epoch)), value)
        self.__current_epoch += 1

    def __save_parameters(self, params: SimulationParameters) -> None:
        params_dict = dataclasses.asdict(params)

        for key, value in params_dict.items():
            if type(value) is np.ndarray:
                params_dict[key] = params_dict[key].tolist()

        with open(os.path.join(self.__out_folder_path, config.PARAMS_FILENAME), "w") as file:
            json.dump(params_dict, file, default=lambda o: o.__dict__, sort_keys=True, indent=4)


if __name__ == '__main__':
    default_params = SimulationParameters()
    Saver("out", default_params)
