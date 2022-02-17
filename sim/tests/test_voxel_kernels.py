import pytest
from sim.src.sph.kernels.voxel_kernels import random_samples, get_index
from numba import cuda
import numba.cuda.random as random

import numpy as np

@cuda.jit 
def random_samples_call(samples, rng_states, low, high):
    i = get_index()
    random_samples(samples, len(samples), low, high, rng_states, i)

@pytest.mark.parametrize("ranges", [(1, 15), (0, 3), (2, 4)])
def test_random_samples(ranges):
    low, high = ranges
    samples = np.zeros(10, dtype=np.int32)
    d_samples = cuda.to_device(samples)
    rng_states = random.create_xoroshiro128p_states(1, seed=121)
    random_samples_call[1, 1](d_samples, rng_states, low, high)
    cuda.synchronize()
    result = d_samples.copy_to_host()
    assert np.all(result < high), f"Given sample exceed upper bound: {result}"
    assert np.all(result >= low), f"Given sample exceed lower bound: {result}"