import numpy as np


class FPSCounter:
    def __init__(self, samples=10):
        self.samples = samples
        self.last_fps = 0
        self.__samples_taken = []

    def get_fps(self, delta_time):
        self.__samples_taken.append(delta_time)
        if len(self.__samples_taken) >= self.samples:
            avg_delta = np.sum(self.__samples_taken) / len(self.__samples_taken)
            self.last_fps = int(1 / avg_delta)

        return self.last_fps
