import unittest


from common.data_classes import Segment, Pipe
from common.pipe_builder import PipeBuilder

import numpy as np


class PipeTests(unittest.TestCase):

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


    def test_pipe_transform(self):
        pipe = PipeBuilder()\
            .add_increasing_segment(2., 3.)\
            .add_roller_segment(1.)\
            .add_increasing_segment(2., 3.)\
            .get_result()
        pipe.transform(600, 70)

        expected_pipe_numpy = PipeBuilder()\
            .with_starting_position((0., 300., 300.))\
            .with_starting_radius(10.)\
            .with_ending_radius(10.)\
            .with_starting_length(100.)\
            .add_increasing_segment(200., 30.)\
            .add_roller_segment(100.)\
            .add_increasing_segment(200., 30.)\
            .get_result()\
            .to_numpy()

        print(pipe.to_numpy())
        print(expected_pipe_numpy)
        self.assertTrue(np.all(pipe.to_numpy() == expected_pipe_numpy))

if __name__ == '__main__':
    unittest.main()
