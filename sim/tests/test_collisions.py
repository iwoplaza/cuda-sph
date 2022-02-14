import unittest

from sim.src.sph.kernels.base_kernels import *
from common.pipe_builder import PipeBuilder


@cuda.jit
def find_segment_kernel(
        index,
        position,
        pipe
):
    for i in range(0, len(index)):
        index[i] = find_segment(position[i], pipe)


@cuda.jit
def calc_vector_length_kernel(
        lengths,
        vectors
):
    for i in range(len(lengths)):
        lengths[i] = calc_vector_length(vectors[i])


@cuda.jit
def calc_distance_between_points_kernel(
        lengths,
        first,
        second
):
    for i in range(len(lengths)):
        lengths[i] = calc_distance_between_points(first[i], second[i])


@cuda.jit
def calc_x_at_segment_beginning_kernel(
        results,
        pipe,
        pipe_segments
):
    for i in range(len(results)):
        results[i] = calc_x_at_segment_beginning(pipe, pipe_segments[i])


@cuda.jit
def is_out_of_pipe_kernel(
        results,
        positions,
        pipe,
        pipe_segments
):
    for i in range(len(results)):
        results[i] = is_out_of_pipe(positions[i], pipe, pipe_segments[i])


@cuda.jit
def calc_collision_move_vector_length_kernel(
        results,
        l_points,
        l_vectors,
        p_points,
        p_vectors
):
    for i in range(len(results)):
        results[i] = calc_collision_move_vector_length(l_points[i], l_vectors[i], p_points[i], p_vectors[i])


@cuda.jit
def collisions_resolution_kernel(
        results,
        positions,
        speed,
        pipe,
        pipe_segments
):
    i = get_index()
    pipe_segments[i] = find_segment(positions[i], pipe)
    if is_out_of_pipe(positions[i], pipe, pipe_segments[i]):
        solve_collision(positions[i], speed[i], pipe, pipe_segments[i])
    results[i] = is_out_of_pipe(positions[i], pipe, pipe_segments[i])


@cuda.jit
def test_rand_position_inside_pipe_kernel(
        result,
        positions,
        pipe,
        rng_states
):
    i = get_index()
    if i >= len(result):
        return

    rand_position_inside_pipe(positions[i], pipe, rng_states, i)
    pipe_segment = find_segment(positions[i], pipe)
    result[i] = is_out_of_pipe(positions[i], pipe, pipe_segment)


class CollisionTests(unittest.TestCase):

    def test_find_segment(self):
        position = np.array([[-0.1, 0, 2], [0.5, 3, 4], [1, 0, 0], [2.5, 0, -5], [5, 5, 5]], dtype=np.float64)
        pipe = PipeBuilder().add_roller_segment(1.0)\
            .add_increasing_segment(1.0, 3.0)\
            .get_result()\
            .to_numpy()
        result = np.array([-2, -2, -2, -2, -2], dtype=np.float64)

        d_pipe = cuda.to_device(pipe)
        d_position = cuda.to_device(position)
        d_result = cuda.to_device(result)

        find_segment_kernel[1, 1](d_result, d_position, d_pipe)
        cuda.synchronize()
        result = d_result.copy_to_host()
        expected = np.array([-1, 0, 1, 2, -1], dtype=np.float64)
        self.assertTrue(np.all(result == expected))

    def test_calc_vector_length(self):
        vectors = np.array([[0, 2, 2], [1, 1, 1]], dtype=np.float64)
        result = np.array([-1, -1], dtype=np.float64)

        d_vectors = cuda.to_device(vectors)
        d_result = cuda.to_device(result)
        calc_vector_length_kernel[1, 1](d_result, d_vectors)
        cuda.synchronize()
        result = d_result.copy_to_host()
        expected = np.array([8**0.5, 3**0.5], dtype=np.float64)
        self.assertTrue(np.all(result == expected))

    def test_calc_distance(self):
        first = np.array([[1, 1, 1], [4, 5, 6]], dtype=np.float64)
        second = np.array([[2, 3, 5], [-2, 5, 1]], dtype=np.float64)
        result = np.array([-1, -1], dtype=np.float64)

        d_first = cuda.to_device(first)
        d_second = cuda.to_device(second)
        d_result = cuda.to_device(result)

        calc_distance_between_points_kernel[1, 1](d_result, d_first, d_second)
        cuda.synchronize()
        result = d_result.copy_to_host()
        print(result)
        expected = np.array([21**0.5, 61**0.5], dtype=np.float64)
        print(expected)
        self.assertTrue(np.all(result == expected))

    def test_calc_x_at_segment_beginning(self):
        pipe = PipeBuilder()\
            .add_increasing_segment(4, 1) \
            .add_lessening_segment(6, 1.5) \
            .get_result() \
            .to_numpy()
        pipe_segments = np.array([0, 1, 2])

        d_pipe = cuda.to_device(pipe)
        d_pipe_segments = cuda.to_device(pipe_segments)
        d_result = cuda.to_device(np.array([-1, -1, -1], dtype=np.float64))

        calc_x_at_segment_beginning_kernel[1, 1](d_result, d_pipe, d_pipe_segments)
        cuda.synchronize()
        result = d_result.copy_to_host()
        print(result)
        expected = np.array([0, 1, 5], dtype=np.float64)
        print(expected)
        self.assertTrue(np.all(result == expected))

    def test_is_out_of(self):
        positions = np.array([[0.5, 0.5, 0.5], [0.76, 0.9, 0.9], [0.01, 1, 0.5],
                              [1.25, 0.8, 0.8], [1.5, 1.25, 1.40], [1.99, 1.3, 1.6],
                              [2.2, 1, 1], [2.6, 1, 1], [2.95, 0.2, 0.2]], dtype=np.float64)
        pipe = PipeBuilder().add_increasing_segment(1, 1)\
            .add_lessening_segment(1, 1.5)\
            .get_result()\
            .to_numpy()
        segments = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        results = np.array([False]*9, dtype=bool)

        d_positions = cuda.to_device(positions)
        d_pipe = cuda.to_device(pipe)
        d_segments = cuda.to_device(segments)
        d_result = cuda.to_device(results)

        is_out_of_pipe_kernel[1, 1](d_result, d_positions, d_pipe, d_segments)
        cuda.synchronize()
        result = d_result.copy_to_host()
        print(d_positions.copy_to_host())
        print(result)
        expected = np.array([False, True, True,
                             False, True, True,
                             False, True, False], dtype=np.bool_)
        print(expected)
        self.assertTrue(np.all(result == expected))

    def test_calc_collision_move_vector_length(self):
        l_points = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float64)
        l_vectors = np.array([[3, 2, 1], [0, 0, 4]], dtype=np.float64)
        p_points = np.array([[8, -5, 2], [4, 1, 5]], dtype=np.float64)
        p_vectors = np.array([[-4, 9, 2], [-3, 0, 0]], dtype=np.float64)
        result = np.array([0, 0], dtype=np.float64)

        d_l_points = cuda.to_device(l_points)
        d_l_vectors = cuda.to_device(l_vectors)
        d_p_points = cuda.to_device(p_points)
        d_p_vectors = cuda.to_device(p_vectors)
        d_result = cuda.to_device(result)

        calc_collision_move_vector_length_kernel[1, 1](d_result, d_l_points, d_l_vectors, d_p_points, d_p_vectors)
        cuda.synchronize()
        result = d_result.copy_to_host()
        print(result)
        expected = np.array([1, 1], dtype=np.float64)
        print(expected)
        self.assertTrue(np.all(result == expected))

    def test_collision_resolution(self):
        N = 1
        positions = np.array([[1.3, 1.5/2**0.5, 1.5/2**0.5]], dtype=np.float64)
        speed = np.array([[-0.5, -0, 0.]], dtype=np.float64)
        pipe = PipeBuilder().add_increasing_segment(1, 1)\
            .add_lessening_segment(2, 1.5)\
            .get_result()\
            .to_numpy()
        pipe_segments = np.array([-2]*N, dtype=np.int32)
        result = np.array([False]*N, dtype=np.bool_)
        print(pipe)
        d_positions = cuda.to_device(positions)
        d_speed = cuda.to_device(speed)
        d_pipe = cuda.to_device(pipe)
        d_pipe_segments = cuda.to_device(pipe_segments)
        d_result = cuda.to_device(result)

        collisions_resolution_kernel[1, N](d_result, d_positions, d_speed, d_pipe, d_pipe_segments)
        cuda.synchronize()
        result = d_result.copy_to_host()
        print(speed[0])
        print(d_speed.copy_to_host()[0])
        print(positions[0])
        print(d_positions.copy_to_host()[0])

        print(result)
        self.assertFalse(np.all(result))

    def test_rand_position_inside_pipe(self):
        N = 16
        positions = np.random.rand(N, 3).astype(np.float64)*3
        pipe = PipeBuilder().add_increasing_segment(1, 1)\
            .add_lessening_segment(1, 1.5)\
            .get_result()\
            .to_numpy()
        result = np.array([False]*N, dtype=np.bool_)
        d_positions = cuda.to_device(positions)
        d_pipe = cuda.to_device(pipe)
        d_result = cuda.to_device(result)
        rng_states = random.create_xoroshiro128p_states(N, seed=43)

        test_rand_position_inside_pipe_kernel[1, N](d_result, d_positions, d_pipe, rng_states)
        cuda.synchronize()
        result = d_result.copy_to_host()

        self.assertFalse(np.all(result))


if __name__ == '__main__':
    unittest.main()
