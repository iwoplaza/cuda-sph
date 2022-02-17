import pytest
from sim.src.sph.kernels.voxel_kernels import random_samples, get_index, get_voxel_id_from_acc
from numba import cuda
import numba.cuda.random as random

import numpy as np

@cuda.jit 
def random_samples_call(samples, rng_states, low, high):
    i = get_index()
    random_samples(samples, len(samples), low, high, rng_states, i)

@cuda.jit
def get_voxel_acc_call(result, accum_arr, p_idx):
    i = get_index()
    if i == 0:
        result[0] = get_voxel_id_from_acc(accum_arr, p_idx)

@pytest.mark.parametrize("ranges", [(1, 15), (0, 3), (2, 4)])
def test_random_samples(ranges):
    low, high = ranges
    samples = np.zeros(10, dtype=np.int32)
    d_samples = cuda.to_device(samples)
    rng_states = random.create_xoroshiro128p_states(1, seed=121)
    for i in range(1000):
        random_samples_call[1, 1](d_samples, rng_states, low, high)
        cuda.synchronize()
        result = d_samples.copy_to_host()
        assert np.all(result < high), f"Given sample exceed upper bound: {result}"
        assert np.all(result >= low), f"Given sample exceed lower bound: {result}"

@pytest.mark.parametrize("p_idx, expected_voxel", [(10, 1), (12, 4), (11, 1), (15, 6), (14, 4)])
def test_get_voxel_id_from_acc(p_idx, expected_voxel):
    accum_arr = np.array([10, 12, 12, 12, 15, 15])
    d_accum_arr = cuda.to_device(accum_arr)
    voxel_idx = np.zeros(1, dtype=np.int32)
    d_voxel_idx = cuda.to_device(voxel_idx)
    get_voxel_acc_call[1, 1](d_voxel_idx, d_accum_arr, p_idx)
    cuda.synchronize()

    result = d_voxel_idx.copy_to_host()
    assert result == expected_voxel, f"Wrong voxel idx, expected {expected_voxel}, got {result}"