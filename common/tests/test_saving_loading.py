import unittest
import shutil
import numpy as np
from numpy import array

from common.data_classes import SimulationState, SimulationParameters, Segment
from common.serializer.loader import Loader
from common.serializer.saver import Saver


def generate_simulation_state(params: SimulationParameters) -> SimulationState:
    simulation_state = SimulationState(array(1), array(1), array(1), array(1))
    simulation_state.set_random_from_params(params)
    for name in vars(simulation_state).keys():
        setattr(simulation_state, name, np.random.rand(params.n_particles, 3))
    return simulation_state


class MyTestCase(unittest.TestCase):
    def test_settings_loading(self):
        path = "data"
        parameters = SimulationParameters()
        second_segment = Segment((1.0, 0.0, 0.0), 1, 4, 5)
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
        np.random.seed = 43

        first_state = generate_simulation_state(parameters)
        second_state = generate_simulation_state(parameters)

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
