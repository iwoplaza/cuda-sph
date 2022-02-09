import math


class ThreadOrganizer:
    def __init__(self):
        self.__detect_devices()

    # TODO: use device info to compute this
    def organize(self, n_threads_total) -> tuple[int, int]:
        threads_per_grid: int = 64
        grids_per_block: int = math.ceil(n_threads_total / threads_per_grid)
        return threads_per_grid, grids_per_block

    # TODO:
    def __detect_devices(self):
        pass
