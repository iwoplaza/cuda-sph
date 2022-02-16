import os
from dataclasses import fields
import numpy as np
import json
import config
from common.data_classes import SimulationState, SimulationParameters, Pipe, Segment


class Loader:
    """
    Class responsible for loading saved simulation frames
    """

    def __init__(self, out_dirname: str) -> None:
        """
        :param out_dirname: path to folder with data about simulation
        """
        self.__out_folder_path = os.path.join(config.PROJECT_DIRNAME, out_dirname)
        if not os.path.exists(self.__out_folder_path):
            raise Exception(f'Directory ({self.__out_folder_path}) does not exists! '
                            f'Could not load simulation!')
        with open(os.path.join(self.__out_folder_path, config.PARAMS_FILENAME)) as file:
            self.__json_object = json.load(file)

    def load_simulation_parameters(self) -> SimulationParameters:
        """
        :return: Loaded SimulationParameters
        """
        params_dict = dict()
        for field in fields(SimulationParameters):
            if field.name == 'pipe':
                pipe = self.__load_pipe()
                params_dict['pipe'] = pipe
            elif field.type == 'np.ndarray':
                params_dict[field.name] = np.asarray(self.__json_object[field.name])
            else:
                params_dict[field.name] = self.__json_object[field.name]
        return SimulationParameters(**params_dict)

    def __load_pipe(self) -> Pipe:
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
        state_dict = dict()
        for field in fields(SimulationState):
            state_dict[field.name] = np.load(
                os.path.join(self.__out_folder_path, field.name + "_" + str(epoch)) + ".npy"
            )
        return SimulationState(**state_dict)
