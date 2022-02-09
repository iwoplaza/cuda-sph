import numpy as np
import json
from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters, Pipe, Segment


SETTINGS_FILE = "/settings.json"


class Loader:
    """
    Class responsible for loading saved simulation frames
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param folder_path: path to folder with data about simulation
        """
        self.__folder_path = folder_path
        with open(folder_path + SETTINGS_FILE) as file:
            self.__json_object = json.load(file)

    def load_simulation_parameters(self) -> SimulationParameters:
        """
        :return: Loaded SimulationParameters
        """
        # print("Loading")
        simulation_parameters = SimulationParameters()
        for name, value in vars(SimulationParameters()).items():
            self.__manage_element(simulation_parameters, name, value)
        # print(simulation_parameters)
        return simulation_parameters

    def __manage_element(self, simulation_parameters: SimulationParameters, name: str, value) -> None:
        """
        Loads from json or from .npy file.

        :param simulation_parameters: Parameters of simulation
        :param name: Name of SimulationParameters variable
        :param value: Value of SimulationParameters variable
        """
        if type(value) is np.ndarray:
            setattr(simulation_parameters, name, np.load(self.__folder_path + "/" + name + ".npy"))
        elif name == 'pipe':
            pipe = self.load_pipe()
            setattr(simulation_parameters, 'pipe', pipe)
        else:
            setattr(simulation_parameters, name, self.__json_object[name])

    def load_pipe(self) -> Pipe:
        segments_data = self.__json_object["pipe"]['segments']
        segments = []
        for segment_data in segments_data:
            segments.append(Segment(
                start_point=tuple(segment_data["start_point"]),
                start_radius=segment_data["start_radius"],
                end_radius=segment_data["end_radius"],
                length=segment_data["length"]
            ))
        return Pipe(segments)

    def load_simulation_state(self, epoch: int) -> SimulationState:
        """
        Load simulation state for selected epoch

        :param epoch: Number of epoch to load data from
        :return: SimulationState for selected epoch
        """
        state = SimulationState()
        for name in vars(state).keys():
            setattr(state, name, np.load(self.__folder_path + "/" + name + "_" + str(epoch) + ".npy"))
        return state
