import unittest
import shutil
import numpy as np

from common.main.serializer.loader import Loader
from common.main.serializer.saver import Saver

from common.main.data_classes.simulation_data_classes import Pipe, Segment, SimulationState, SimulationParameters


def get_simulation_parameters() -> SimulationParameters:
    first_segment = Segment((0, 0, 0), 2, None)
    second_segment = Segment((1, 1, 1), 5, first_segment)

    particle_mass = 0.123
    simulation_duration = 65
    fps = 60
    particles_number = 1000000
    particles_radius = 0.0023
    pipe = Pipe([first_segment, second_segment])
    influence_radius = 0.324
    space_dims = (1, 1, 1)
    voxel_dim = (1, 2, 3)

    return SimulationParameters(particle_mass, simulation_duration, fps, particles_number, particles_radius, pipe,
                                influence_radius, space_dims, voxel_dim)


class MyTestCase(unittest.TestCase):
    def test_settings_loading(self):
        path = "data"
        parameters = get_simulation_parameters()

        Saver(path, parameters)
        loader = Loader(path)
        loader.load_simulation_parameters()
        loaded_parameters = loader.load_simulation_parameters()

        self.assertEqual(loaded_parameters.particle_mass, parameters.particle_mass)
        self.assertEqual(loaded_parameters.simulation_duration, parameters.simulation_duration)
        self.assertEqual(loaded_parameters.fps, parameters.fps)
        self.assertEqual(loaded_parameters.particles_number, parameters.particles_number)
        self.assertEqual(loaded_parameters.particles_radius, parameters.particles_radius)
        self.assertEqual(loaded_parameters.pipe, parameters.pipe)
        self.assertEqual(loaded_parameters.influence_radius, parameters.influence_radius)
        self.assertEqual(loaded_parameters.space_dims, parameters.space_dims)
        self.assertEqual(loaded_parameters.voxel_dim, parameters.voxel_dim)

        shutil.rmtree(path)

    def test_pressure_and_particles(self):
        path = "data"
        parameters = get_simulation_parameters()

        particle_number = 10000
        np.random.seed = 0
        first_pressure = np.random.rand(100, 1)
        second_pressure = np.random.rand(100, 1)
        first_positions = np.random.rand(particle_number, 3)
        second_positions = np.random.rand(particle_number, 6)
        first_velocity = np.random.rand(particle_number, 3)
        second_velocity = np.random.rand(particle_number, 3)

        first_state = SimulationState(first_positions, first_velocity, first_pressure, np.random.rand(1, 1),
                                      np.random.rand(1, 1))
        second_state = SimulationState(second_positions, second_velocity, second_pressure, np.random.rand(1, 1),
                                       np.random.rand(1, 1))

        saver = Saver(path, parameters)
        saver.save_next_state(first_state)
        saver.save_next_state(second_state)

        loader = Loader(path)

        self.assertTrue(np.alltrue(loader.load_simulation_state(0).pressure == first_state.pressure))
        self.assertTrue(np.alltrue(loader.load_simulation_state(0).position == first_state.position))
        self.assertTrue(np.alltrue(loader.load_simulation_state(0).velocity == first_state.velocity))
        self.assertTrue(np.alltrue(loader.load_simulation_state(1).pressure == second_state.pressure))
        self.assertTrue(np.alltrue(loader.load_simulation_state(1).position == second_state.position))
        self.assertTrue(np.alltrue(loader.load_simulation_state(1).velocity == second_state.velocity))

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
