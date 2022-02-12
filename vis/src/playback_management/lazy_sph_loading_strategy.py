from typing import List
import numpy as np

from common.serializer.loader import Loader
from common.data_classes import SimulationParameters
from vis.src.vector import Vec3f
from .loading_strategy import LoadingStrategy


def generate_random_points(amount: int) -> np.ndarray:
    points = []

    x_range = (0.0, 10.0)
    y_range = (0.0, 10.0)
    z_range = (0.0, 10.0)

    for i in range(amount):
        points.append([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range),
        ])

    return np.array(points, np.float64)


def move_points(points: List[Vec3f]):
    for i in range(len(points)):
        points[i] = (
            points[i][0] + np.random.uniform(-1, 1) * 0.01,
            points[i][1] + np.random.uniform(-1, 1) * 0.01,
            points[i][2] + np.random.uniform(-1, 1) * 0.01
        )


class LazySPHLoadingStrategy(LoadingStrategy):
    def __init__(self, folder_path: str):
        self.__loader = Loader(folder_path)

        params = self.__loader.load_simulation_parameters()
        self.__params: SimulationParameters = params

        self.points = generate_random_points(50000)
        self.end_frame = int(params.simulation_duration * self.__params.fps) - 1

        self.latest_frame_index = None
        self.latest_frame_data = None

    def get_data_at_time(self, time: float):
        """
        :param: time in seconds
        """

        epoch = min(int(time * self.__params.fps), self.end_frame)

        if self.latest_frame_index == epoch:
            return self.latest_frame_data

        print(f"Epoch: {epoch + 1}/{self.end_frame + 1}")
        data = self.__loader.load_simulation_state(epoch)
        self.latest_frame_data = data.position
        self.latest_frame_index = epoch

        return self.latest_frame_data

    def get_duration(self):
        return self.__params.simulation_duration
