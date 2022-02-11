import math


MAX_THREADS_PER_BLOCK = 1024
MAX_GRID_SIZE_X = 2_147_483_647
SM_COUNT = 12


def organize(n_threads_total) -> tuple[int, int]:
    if n_threads_total >= MAX_GRID_SIZE_X:
        raise Exception(f"Critical error. Cannot create more than {MAX_GRID_SIZE_X} CUDA threads.")
    block_size = MAX_THREADS_PER_BLOCK
    grid_size = math.ceil(n_threads_total / block_size)
    return grid_size, block_size

