import math


THREADS_PER_BLOCK = 128


class ThreadOrganizer:
    def __init__(self):
        self.__detect_devices()

    # TODO: use device info to compute this
    def organize(self, n_threads_total) -> tuple[int, int]:
        block_size = THREADS_PER_BLOCK
        grid_size = math.ceil(n_threads_total / block_size)
        return grid_size, block_size

    # TODO:
    def __detect_devices(self):
        pass
