import unittest
import shutil
import numpy as np

from common.results_managing.loader import Loader
from common.results_managing.saver import Saver


class MyTestCase(unittest.TestCase):
    def test_settings_loading(self):
        path = "data"
        pipe = [1, 2, 3]
        particle_number = 25439

        Saver(path, pipe, particle_number)
        loader = Loader(path)

        self.assertEqual(loader.get_particles_number(), particle_number)
        self.assertEqual(loader.load_pipe(), pipe)
        shutil.rmtree(path)

    def test_pressure_and_particles(self):
        path = "data"
        pipe = [1, 2, 3]
        particle_number = 100000
        np.random.seed = 0
        first_pressure = np.random.rand(100, 1)
        second_pressure = np.random.rand(particle_number, 6)
        first_particles = np.random.rand(100, 1)
        second_particles = np.random.rand(particle_number, 6)

        saver = Saver(path, pipe, particle_number)
        saver.save_next_epoch(first_particles, first_pressure)
        saver.save_next_epoch(second_particles, second_pressure)

        loader = Loader(path)

        self.assertTrue(np.alltrue(loader.load_pressure_data(0) == first_pressure))
        self.assertTrue(np.alltrue(loader.load_pressure_data(1) == second_pressure))
        self.assertTrue(np.alltrue(loader.load_particles_state(0) == first_particles))
        self.assertTrue(np.alltrue(loader.load_particles_state(1) == second_particles))

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
