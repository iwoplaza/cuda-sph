import unittest


from common.data_classes import Segment, Pipe
from common.pipe_builder import PipeBuilder

import numpy as np


class MyTestCase(unittest.TestCase):

    def test_segment_to_numpy(self):
        segment = Segment(start_radius=4.5)

        expected = np.array([0, 0, 0, 4.5, 1])

        self.assertTrue(np.all(segment.to_numpy() == expected))

    def test_simple_pipe_to_numpy(self):
        pipe = Pipe([Segment()])

        expected = np.array([[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]])

        self.assertTrue(np.all(pipe.to_numpy() == expected))

    def test_pipe_to_numpy(self):
        pipe = PipeBuilder().with_starting_radius(3)\
            .add_roller_segment(1)\
            .add_increasing_segment(1, 2)\
            .get_result()

        expected = np.array([[0, 0, 0, 3, 1], [1, 0, 0, 1, 1], [2, 0, 0, 1, 1], [3, 0, 0, 3, 1]])

        self.assertTrue(np.all(pipe.to_numpy() == expected))


if __name__ == '__main__':
    unittest.main()
