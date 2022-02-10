# TODO: signatures of kernels has been changed. tests has to be updated
# TODO 2: actually lots of things changed, tests are outdated
# :(
import math

import pytest
import numpy as np
from numba import cuda
from math import ceil

from common.main.data_classes.simulation_data_classes import SimulationParameters, SimulationState
from sim.src.main.physics import constants
from sim.src.main.physics.sph import VoxelSPHStrategy
from sim.src.main.physics.sph.naive_strategy import kernels


N_ELEMENTS = 1000
BLOCK_SIZE: int = 128
N_BLOCKS: int = ceil(N_ELEMENTS / BLOCK_SIZE)
SPACE_SIZE = 10


@pytest.fixture 
def position_random():
    return np.asarray(
        [np.random.random() * SPACE_SIZE for _ in range(N_ELEMENTS)]
    )


@pytest.fixture 
def velocity_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.float64)


@pytest.fixture 
def pressure_term_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.float64)


@pytest.fixture 
def viscosity_term_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.float64)


@pytest.fixture 
def density_random():
    return np.random.random(N_ELEMENTS).astype(np.float64)


@pytest.fixture 
def viscosity_random():
    return np.random.random(N_ELEMENTS).astype(np.float64)


@pytest.fixture 
def density_random():
    return np.random.random(N_ELEMENTS).astype(np.float64)


def test_density_kernel_update_array(position_random):
    """Density kernel should update density array and shouldn't update position array."""
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(np.zeros(N_ELEMENTS, dtype=np.float64))
    kernels.density_kernel[N_BLOCKS, BLOCK_SIZE](
        d_density,
        d_position,
        constants.MASS,
        constants.INF_R
    )
    h_density = d_density.copy_to_host()
    h_position = d_position.copy_to_host()
    assert np.any(h_density != 0), f"Density array was not updated by the kernels.density_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the kernels.density_kernel")


def test_pressure_kernel_update_array(density_random, position_random):
    """Pressure term kernel should update pressure_term array.
        Position and density arrays shouldn't be modified."""
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(density_random)
    d_pressure_term = cuda.to_device(np.random.random((N_ELEMENTS, 3)))
    kernels.pressure_kernel[N_BLOCKS, BLOCK_SIZE](
        d_pressure_term,
        d_density,
        d_position,
        constants.MASS,
        constants.INF_R,
        constants.K,
        constants.RHO_0
    )
    h_pressure_term = d_pressure_term.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    assert np.any(h_pressure_term != 0), "pressure_term array was not updated by the pressure_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the pressure_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the pressure_kernel")


def test_viscosity_kernel_update_array(density_random, position_random, velocity_random):
    """Viscosity kernel should update viscosity array.
        Position, density and velocity shouldn't be modified."""
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(density_random)
    d_velocity = cuda.to_device(velocity_random)
    d_viscosity = cuda.to_device(np.random.random((N_ELEMENTS, 3)))
    kernels.viscosity_kernel[N_BLOCKS, BLOCK_SIZE](
        d_viscosity,
        d_density,
        d_position,
        d_velocity,
        constants.MASS,
        constants.INF_R,
        constants.VISC
    )
    h_viscosity = d_viscosity.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    h_velocity = d_velocity.copy_to_host()
    assert np.any(h_viscosity != 0), "viscosity array was not updated by the viscosity_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_velocity, velocity_random, err_msg="velocity array was modified by the viscosity_kernel")


def test_organize_voxels():
    """voxels, voxel_particle_map and voxel_begins have to be correct
        after _initialize_computation() call in VoxelSPHStrategy"""
    params = SimulationParameters(n_particles=np.int32(20))
    state = SimulationState()
    state.set_random_from_params(params)
    sph_strategy = VoxelSPHStrategy(params)
    sph_strategy.old_state = state
    sph_strategy._initialize_computation()
    vox_size = params.voxel_size
    space_size = params.space_size
    space_dims = [math.ceil(space_size[i] / vox_size[i]) for i in range(3)]  # num of voxels in each dimension
    width, height, depth = space_dims

    # check if voxels are assigned correctly
    for i in range(params.n_particles):
        pos = position_random[i]
        vox_3d = [pos[i] / space_dims[i] for i in range(3)]
        vox = vox_3d[0] + vox_3d[1] * width + vox_3d[2] * width * height
        assert vox == sph_strategy.voxels[i]


def test_get_neighbours(position_random):
    MAX_NEIGHBOURS = 32
    result_neighbours = np.asarray([-1 for _ in range(MAX_NEIGHBOURS)], dtype=np.int32)


