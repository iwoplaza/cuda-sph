import pytest 
from test import BLOCK_SIZE, N_ELEMENTS, N_BLOCKS
import numpy as np
from src.kernels import density_kernel, pressure_kernel, viscosity_kernel
from numba import cuda 


@pytest.fixture 
def position_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.double)


@pytest.fixture 
def velocity_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.double)


@pytest.fixture 
def pressure_term_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.double)


@pytest.fixture 
def viscosity_term_random():
    return np.random.random((N_ELEMENTS, 3)).astype(np.double)


@pytest.fixture 
def density_random():
    return np.random.random(N_ELEMENTS).astype(np.double)


@pytest.fixture 
def viscosity_random():
    return np.random.random(N_ELEMENTS).astype(np.double)


@pytest.fixture 
def density_random():
    return np.random.random(N_ELEMENTS).astype(np.double)


@pytest.mark.parametrize("mass", [1e-5, 1e-10, 1, 10])
def test_density_kernel_update_array(position_random, mass):
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(np.zeros(N_ELEMENTS, dtype=np.double))
    density_kernel[N_BLOCKS, BLOCK_SIZE](d_density, d_position, mass, 0.1)
    h_density = d_density.copy_to_host()
    h_position = d_position.copy_to_host()
    assert np.any(h_density != 0), f"Density array was not updated by the density_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the density_kernel")


@pytest.mark.parametrize("mass", [1e-5, 1e-10, 1, 10])
def test_pressure_kernel_update_array(density_random, position_random, mass):
    d_position = cuda.to_device(position_random)
    d_density = cuda.to_device(density_random)
    d_pressure_term = cuda.to_device(np.random.random((N_ELEMENTS, 3)))
    pressure_kernel[N_BLOCKS, BLOCK_SIZE](d_density, d_position, d_pressure_term, mass, 0.1, 0.056, 0.2)
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
    viscosity_kernel[N_BLOCKS, BLOCK_SIZE](d_density, d_position, d_velocity, d_viscosity, mass, 0.1, 0.056)
    h_viscosity= d_viscosity.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    h_velocity = d_velocity.copy_to_host()
    assert np.any(h_viscosity != 0), "viscosity array was not updated by the viscosity_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_velocity, velocity_random, err_msg="velocity array was modified by the viscosity_kernel")


