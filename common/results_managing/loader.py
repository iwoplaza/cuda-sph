import numpy as np
import json
import common.results_managing.constants as constants

from sim.src.utils import SimulationState, SimulationParameters, Pipe, Segment


class Loader:
    """
    Class responsible for loading saved simulation frames
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param folder_path: path to folder with data about simulation
        """
        self.__folder_path = folder_path
        with open(folder_path + constants.SETTINGS_FILE) as file:
            self.__json_object = json.load(file)

    def load_simulation_state(self, epoch: int) -> SimulationState:
        """
        Load simulation state for selected epoch

        :param epoch: Number of epoch to load data from
        :return: Numpy matrix with information about positions and velocity of particles
        """
        return SimulationState(position=np.load(self.__folder_path + constants.POSITION_FILE % epoch),
                               velocity=np.load(self.__folder_path + constants.VELOCITY_FILE % epoch),
                               pressure=np.load(self.__folder_path + constants.PRESSURE_FILE % epoch),
                               density=None, voxels=None)

    def load_simulation_parameters(self) -> SimulationParameters:
        """
        :return: Loaded SimulationParameters
        """
        return SimulationParameters(fps=self.__json_object[constants.FPS],
                                    influence_radius=self.__json_object[constants.INFLUENCE_RADIUS],
                                    particle_mass=self.__json_object[constants.PARTICLE_MASS],
                                    space_dims=tuple(self.__json_object[constants.SPACE_DIMS]),
                                    voxel_dim=tuple(self.__json_object[constants.VOXEL_DIM]),
                                    particles_number=self.__json_object[constants.PARTICLES_NUMBER],
                                    particles_radius=self.__json_object[constants.PARTICLES_RADIUS],
                                    simulation_duration=self.__json_object[constants.SIMULATION_DURATION],
                                    pipe=self.__load_pipe())

    def __load_pipe(self) -> Pipe:
        prev_segment = None
        segments = []
        for segment_data in self.__json_object[constants.PIPE]:
            print(segment_data)
            radius = segment_data[1]
            end_point = tuple(segment_data[0])
            print("END point: ", end_point)
            segment = Segment(end_point=end_point, radius=radius, prev_segment=prev_segment)
            prev_segment = segment
            segments.append(segment)
        return Pipe(segments)
