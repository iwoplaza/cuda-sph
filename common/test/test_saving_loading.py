import unittest
import shutil
import numpy as np

from common.main.data_classes.simulation_data_classes import SimulationState, SimulationParameters, Segment, get_default_start_sim_state
from common.main.serializer.loader import Loader
from common.main.serializer.saver import Saver


def generate_simulation_state(n_particles: int) -> SimulationState:
    simulation_state = get_default_start_sim_state(n_particles)
    for name in vars(simulation_state).keys():
        setattr(simulation_state, name, np.random.rand(n_particles, 3))
    return simulation_state


class MyTestCase(unittest.TestCase):
    def test_settings_loading(self):
        path = "data"
        parameters = SimulationParameters()
        second_segment = Segment(end_point=(21, 37, 00), radius=2, prev_segment=parameters.pipe.segments[0])
        parameters.pipe.segments.append(second_segment)
        parameters.fps = 40

        Saver(path, parameters)
        loader = Loader(path)
        loaded_parameters = loader.load_simulation_parameters()

        for name in vars(parameters):
            true = getattr(parameters, name)
            expected = getattr(loaded_parameters, name)
            self.assertEqual(type(true), type(expected))
            if type(true) is np.ndarray:
                self.assertTrue(np.all(true == expected))
            else:
                self.assertEqual(true, expected)

        shutil.rmtree(path)

    def test_pressure_and_particles(self):
        path = "data"
        parameters = SimulationParameters()

        n_particles = 10000

        np.random.seed = 43

        first_state = generate_simulation_state(n_particles)
        second_state = generate_simulation_state(n_particles)

        saver = Saver(path, parameters)
        saver.save_next_state(first_state)
        saver.save_next_state(second_state)

        loader = Loader(path)
        loaded_first_state = loader.load_simulation_state(0)
        loaded_second_state = loader.load_simulation_state(1)

        for true, loaded in zip([first_state, second_state], [loaded_first_state, loaded_second_state]):
            for name in vars(first_state).keys():
                true_array = getattr(true, name)
                loaded_array = getattr(loaded, name)
                self.assertTrue(np.all(true_array == loaded_array))

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
