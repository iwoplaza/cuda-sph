import math

from numba import cuda

"""Maximum thread count pee grid"""
MAX_GRID_SIZE_X = 2_147_483_647

"""Compute capability to core count mapping"""
core_count = {
    (2, 0): 32,
    (2, 1): 48,
    (3, 0): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,
    (5, 2): 128,
    (6, 0): 64,
    (6, 1): 128,
    (7, 0): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128
}


def organize(n_threads_total) -> tuple[int, int]:
    device = cuda.get_current_device()
    block_size = core_count[device.compute_capability]
    if n_threads_total >= MAX_GRID_SIZE_X:
        raise Exception(f"Critical error. Cannot create more than {MAX_GRID_SIZE_X} CUDA threads.")
    grid_size = math.ceil(n_threads_total / block_size)
    return grid_size, block_size