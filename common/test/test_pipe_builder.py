import unittest

from common.main.data_classes.simulation_data_classes import Segment, Pipe
from common.main.data_classes.pipe_builder import PipeBuilder


class MyTestCase(unittest.TestCase):

    def test_basic_builder(self):
        pipe = PipeBuilder().get_result()

        self.assertEqual(Pipe([Segment()]), pipe)

    def test_setting_first_segment(self):
        pipe = PipeBuilder().with_starting_position((1, 2, 3))\
            .with_starting_length(20)\
            .with_starting_radius(4)\
            .with_ending_radius(5)\
            .get_result()
        first_segment = Segment(start_point=(1, 2, 3), start_radius=4, end_radius=5, length=20)
        expected_pipe = Pipe([first_segment])

        self.assertEqual(expected_pipe, pipe)

    def test_length_error_first_segment(self):
        with self.assertRaises(AssertionError) as error:
            PipeBuilder().with_starting_length(-10).get_result()

        self.assertEqual(PipeBuilder._NEGATIVE_LENGTH_MESSAGE, str(error.exception))

    def test_radius_error(self):
        with self.assertRaises(AssertionError) as error:
            PipeBuilder().with_starting_radius(-1).get_result()

        self.assertEqual(PipeBuilder._NEGATIVE_RADIUS_MESSAGE, str(error.exception))

        with self.assertRaises(AssertionError) as error:
            PipeBuilder().with_ending_radius(-1).get_result()

        self.assertEqual(PipeBuilder._NEGATIVE_RADIUS_MESSAGE, str(error.exception))

    def test_changing_first_segment_after_adding_others(self):
        with self.assertRaises(AssertionError) as error:
            PipeBuilder().add_roller_segment(20).with_starting_radius(1).get_result()

        self.assertEqual(PipeBuilder._FIRST_SEGMENT_MESSAGE, str(error.exception))

    def test_adding_roller_segment(self):
        pipe = PipeBuilder()\
            .with_ending_radius(4)\
            .add_roller_segment(30)\
            .get_result()

        first_segment = Segment(end_radius=4)
        second_segment = Segment((1, 0, 0), 4, 4, 30)
        expected_pipe = Pipe([first_segment, second_segment])

        self.assertEqual(expected_pipe, pipe)

    def test_adding_lessening_segment(self):
        pipe = PipeBuilder() \
            .with_ending_radius(7) \
            .add_lessening_segment(length=1, change=6) \
            .get_result()

        first_segment = Segment(end_radius=7)
        second_segment = Segment((1, 0, 0), 7, 1, 1)
        expected_pipe = Pipe([first_segment, second_segment])

        self.assertEqual(expected_pipe, pipe)

    def test_adding_increasing_segment(self):
        pipe = PipeBuilder() \
            .with_ending_radius(5) \
            .add_increasing_segment(length=1, change=5) \
            .get_result()

        first_segment = Segment(end_radius=5)
        second_segment = Segment((1, 0, 0), 5, 10, 1)
        expected_pipe = Pipe([first_segment, second_segment])

        self.assertEqual(expected_pipe, pipe)

    def test_adding_many_segments(self):
        pipe = PipeBuilder() \
            .with_starting_radius(10) \
            .with_ending_radius(5) \
            .add_roller_segment(length=1) \
            .add_increasing_segment(1, 5) \
            .get_result()

        first_segment = Segment(start_radius=10, end_radius=5)
        second_segment = Segment((1, 0, 0), 5, 5, 1)
        third_segment = Segment((2, 0, 0), 5, 10, 1)
        expected_pipe = Pipe([first_segment, second_segment, third_segment])

        self.assertEqual(expected_pipe, pipe)

    def test_negative_change_error(self):
        with self.assertRaises(AssertionError) as error:
            PipeBuilder().add_lessening_segment(1, -10).get_result()

        self.assertEqual(PipeBuilder._NEGATIVE_CHANGE_MESSAGE, str(error.exception))

    def test_negative_end_after_lessening_error(self):
        with self.assertRaises(AssertionError) as error:
            PipeBuilder().add_lessening_segment(1, -1).get_result()

        self.assertTrue(type(error.exception) is AssertionError)


if __name__ == '__main__':
    unittest.main()
