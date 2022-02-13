import pytest
import numpy as np
from numba import cuda
import config
from config import INF_R as H
from common.data_classes import SimulationParameters, SimulationState
from sim.src.sph import VoxelSPHStrategy
from sim.src.sph.kernels import naive_kernels as naive_kernels
from sim.src.sph.kernels.voxel_kernels import get_neighbours
from sim.src.sph.thread_layout import organize


N_ELEMENTS = 1000
GRID_SIZE, BLOCK_SIZE = organize(N_ELEMENTS)
SPACE_SIZE = config.DEFAULT_SPACE_SIZE


@pytest.fixture
def position_random():
    return np.asarray(
        [np.random.random() * SPACE_SIZE[i % 3] for i in range(N_ELEMENTS * 3)]
    ).reshape((N_ELEMENTS, 3)).astype(np.float64)


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
    naive_kernels.density_kernel[GRID_SIZE, BLOCK_SIZE](
        d_density,
        d_position,
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
    naive_kernels.pressure_kernel[GRID_SIZE, BLOCK_SIZE](
        d_pressure_term,
        d_density,
        d_position,
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
    naive_kernels.viscosity_kernel[GRID_SIZE, BLOCK_SIZE](
        d_viscosity,
        d_density,
        d_position,
        d_velocity,
    )
    h_viscosity = d_viscosity.copy_to_host()
    h_position = d_position.copy_to_host()
    h_density = d_density.copy_to_host()
    h_velocity = d_velocity.copy_to_host()
    assert np.any(h_viscosity != 0), "viscosity array was not updated by the viscosity_kernel"
    np.testing.assert_array_equal(h_position, position_random, err_msg="position array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_density, density_random, err_msg="density array was modified by the viscosity_kernel")
    np.testing.assert_array_equal(h_velocity, velocity_random, err_msg="velocity array was modified by the viscosity_kernel")
