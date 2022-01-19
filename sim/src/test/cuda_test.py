import sys
import time
import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit
def cuda_add_arrays(A, B, C, i):
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x

    idx = block_width * block_idx + th_idx

    if idx < A.shape[0]:
        C[idx] = A[idx] + B[idx]
        C[idx] *= i


def cpu_add_arrays(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] + B[i]


def test(A, B, C):
    for i in range(A.shape[0]):
        if not C[i] == A[i] + B[i]:
            return False
    return True


if __name__ == 'main':
    N_ELEMENTS = int(3e6)
    print(f"Tests for {N_ELEMENTS} elements arrays")

    # generate arrays on cpu
    h_A = np.random.randint(0, 127, N_ELEMENTS)
    h_B = np.random.randint(0, 127, N_ELEMENTS)
    h_C = np.zeros(N_ELEMENTS, dtype=int)
    # send arrays to gpu
    d_A = cp.array(h_A)
    d_B = cp.array(h_B)
    d_C = cp.array(h_C)
    print(f"Memory size sent to GPU: "
          f"{(sys.getsizeof(h_A) + sys.getsizeof(h_A) + sys.getsizeof(h_A)) * 1e-9}"
          f" GB")

    for i in range(1, 9):
        n_threads_per_block = i * 128
        n_blocks = int(np.ceil(N_ELEMENTS / n_threads_per_block))

        m = 0

        start = time.time_ns()
        cuda_add_arrays[n_blocks, n_threads_per_block](d_A, d_B, d_C, m)
        print(np.asarray(d_C), "yo")
        result = round((time.time_ns() - start) * 1e-9, 5)
        # print((n_threads_per_block, n_blocks, result))