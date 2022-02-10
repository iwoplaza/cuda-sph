# TODO: signatures of kernels has been changed. tests has to be updated
# TODO 2: actually lots of things changed, tests are outdated
# :(

import pytest
import numpy as np
from numba import cuda
from math import ceil
from common.data_classes import SimulationParameters, SimulationState
from sim.src.constants import INF_R as H
from sim.src.sph import VoxelSPHStrategy
from sim.src import constants
from sim.src.sph.kernels import naive_kernels as naive_kernels
from sim.src.sph.kernels.voxel_kernels import get_neighbours


N_ELEMENTS = 1000
BLOCK_SIZE: int = 128
N_BLOCKS: int = ceil(N_ELEMENTS / BLOCK_SIZE)
SPACE_SIZE = 10


@pytest.fixture
def position_random():
    return np.asarray(
        [np.random.random() * SPACE_SIZE for _ in range(N_ELEMENTS * 3)]
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
    naive_kernels.density_kernel[N_BLOCKS, BLOCK_SIZE](
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
    naive_kernels.pressure_kernel[N_BLOCKS, BLOCK_SIZE](
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
    naive_kernels.viscosity_kernel[N_BLOCKS, BLOCK_SIZE](
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


@pytest.fixture
def sph():
    """Fixture providing VoxelSPHStrategy object after with voxels computed"""
    N_PARTICLES = 3
    SPACE_SIZE = (10, 10, 10)
    VOXEL_SIZE = (1, 1, 1)
    params = SimulationParameters(
        n_particles=np.int32(N_PARTICLES),
        space_size=np.asarray(SPACE_SIZE, dtype=np.float64),
        voxel_size=np.asarray(VOXEL_SIZE, dtype=np.float64)
    )
    state = SimulationState(
        position=np.asarray([0, 0, 0,           # voxel 3d (0, 0, 0)     1d 0
                             1.5, 1.5, 1.5,     # voxel 3d (1, 1, 1)     1d 111
                             9.1, 1.1, 5.5],    # voxel 3d (9, 1, 5)     1d 519
                            dtype=np.float64).reshape((N_PARTICLES, 3))
    )
    sph_strategy = VoxelSPHStrategy(params)
    sph_strategy.old_state = state
    sph_strategy._initialize_computation()
    return sph_strategy


def test_assign_voxels_to_particles(sph):
    """Proper voxels should be assigned to particles"""
    voxels = sph.voxels
    assert voxels[0] == 0
    assert voxels[1] == 111
    assert voxels[2] == 519


def test_create_sorted_particle_voxel_map(sph):
    """Particle-voxel map should be created, should contain pairs and should be sorted"""
    voxel_particle_map = sph.voxel_particle_map
    n_particles = sph.params.n_particles

    # check if there are all particles inside map
    present = [False for _ in range(n_particles)]
    for entry in voxel_particle_map:
        particle = entry[1]
        present[particle] = True
    assert False not in present

    # check if map is sorted
    for i in range(n_particles):
        if i < n_particles - 1:
            assert voxel_particle_map[i][0] <= voxel_particle_map[i + 1][0]


def test_voxel_begins(sph):
    """Voxel begins should contains first index of a particle of given voxel in voxel_particle_map"""
    for v in range(len(sph.voxel_begin)):
        begin = sph.voxel_begin[v]
        if begin == -1:
            continue
        assert sph.voxel_particle_map[begin][0] == v
        if v == 0:
            continue
        for idx in range(begin):
            assert sph.voxel_particle_map[idx][0] != v


@pytest.fixture
def sph2():
    """Fixture providing VoxelSPHStrategy object after with voxels computed
        Some particles are neighbours"""
    N_PARTICLES = 5
    params = SimulationParameters(n_particles=np.int32(N_PARTICLES))
    state = SimulationState(
        position=np.asarray([0., 0., 0.,            # not neighbour of p
                             H/4, H/4, H/4,         # not neighbour of p
                             H/2, H/2, H/2,         # neighbour of p
                             H, H, H,               # p
                             3/2*H, 3/2*H, 3/2*H],  # neighbour of p
                            dtype=np.float64).reshape((N_PARTICLES, 3))
    )
    sph_strategy = VoxelSPHStrategy(params)
    sph_strategy.old_state = state
    sph_strategy._initialize_computation()
    return sph_strategy


@cuda.jit
def get_neighbours_kernel(
        neigh_count,
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map
):
    neigh_count[0] = get_neighbours(
        neighbours,
        p_idx,
        position,
        voxel_size,
        space_dim,
        voxel_begin,
        voxel_particle_map
    )


def test_get_neighbours(sph2):
    """Particle should be neighbours with another one iff
        distance between them is less than INF_R"""
    MAX_NEIGHBOURS = 32
    d_neigh_count = cuda.to_device(np.asarray([-1], dtype=np.int32))
    neighbours = np.asarray([-1 for _ in range(MAX_NEIGHBOURS)], dtype=np.int32)
    d_neighbours = cuda.to_device(neighbours)
    space_dim = np.asarray(
        [np.int32(sph2.params.space_size[dim] / sph2.params.voxel_size[dim])
         for dim in range(3)],
        dtype=np.int32
    )
    get_neighbours_kernel[1, 1](
        d_neigh_count,
        d_neighbours,
        3,
        sph2.d_position,
        cuda.to_device(sph2.params.voxel_size),
        cuda.to_device(space_dim),
        sph2.d_voxel_begin,
        sph2.d_voxel_particle_map
    )
    neigh_count = d_neigh_count.copy_to_host()[0]
    neighbours = d_neighbours.copy_to_host()
    # print("\nNeighbours: ", neighbours.tolist())
    assert neigh_count == 3
    assert 0 not in neighbours
    assert 1 not in neighbours
    assert 2 in neighbours[:neigh_count]
    assert 3 in neighbours[:neigh_count]
    assert 4 in neighbours[:neigh_count]
    for i in range(neigh_count, MAX_NEIGHBOURS):
        assert neighbours[i] == -1
