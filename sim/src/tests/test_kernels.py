# TODO: signatures of kernels has been changed. tests has to be updated
# TODO 2: actually lots of things changed, tests are outdated
# :(


import pytest
import numpy as np
from numba import cuda
from math import ceil
from sim.src.main.physics import constants
from sim.src.main.physics.sph.naive_strategy import kernels


N_ELEMENTS = 1000
BLOCK_SIZE: int = 64
N_BLOCKS: int = ceil(N_ELEMENTS / BLOCK_SIZE)


@pytest.fixture 
def position_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.float64)


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


@pytest.mark.parametrize("mass", [1e-5, 1e-10, 1, 10])
def test_density_kernel_update_array(position_random, mass):
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(np.zeros(N_ELEMENTS, dtype=np.float64))
    kernels.density_kernel[N_BLOCKS, BLOCK_SIZE](d_density, d_position, mass, 0.1)
    h_density = d_density.copy_to_host()
    h_position = d_position.copy_to_host()
    assert np.any(h_density != 0), f"Density array was not updated by the kernels.density_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the kernels.density_kernel")


@pytest.mark.parametrize("mass", [1e-5, 1e-10, 1, 10])
def test_pressure_kernel_update_array(density_random, position_random, mass):
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(density_random)
    d_pressure_term = cuda.to_device(np.random.random((N_ELEMENTS, 3)))
    kernels.pressure_kernel[N_BLOCKS, BLOCK_SIZE](d_density, d_position, d_pressure_term, mass, 0.1, 0.056, 0.2)
    h_pressure_term = d_pressure_term.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    assert np.any(h_pressure_term != 0), "pressure_term array was not updated by the pressure_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the pressure_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the pressure_kernel")


@pytest.mark.parametrize("mass", [1e-5, 1e-10, 1, 10])
def test_viscosity_kernel_update_array(density_random, position_random, velocity_random, mass):
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(density_random)
    d_velocity = cuda.to_device(velocity_random)
    d_viscosity = cuda.to_device(np.random.random((N_ELEMENTS, 3)))
    kernels.viscosity_kernel[N_BLOCKS, BLOCK_SIZE](d_viscosity, d_density, d_position, d_velocity, constants.MASS,
                                                   constants.INF_R, constants.VISC)
    h_viscosity= d_viscosity.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    h_velocity = d_velocity.copy_to_host()
    assert np.any(h_viscosity != 0), "viscosity array was not updated by the viscosity_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_velocity, velocity_random, err_msg="velocity array was modified by the viscosity_kernel")


