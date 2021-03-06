from numba import cuda
import numpy as np
import math


@cuda.jit(device=True)
def get_index():
    """Returns a 1-dimensional index of a CUDA thread."""
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    return block_width * block_idx + th_idx


@cuda.jit(device=True)
def subtract(v1, v2, result):
    for dim in range(3):
        result[dim] = v1[dim] - v2[dim]


@cuda.jit(device=True)
def norm_squared(v1, v2):
    """Returns a squared norm (distance) of two vectors.
        Assumes they are the same length."""
    res = 0.0
    for dim in range(len(v1)):
        res += (v1[dim] - v2[dim]) ** 2
    return res


@cuda.jit(device=True)
def norm(v1, v2):
    """Returns a regular norm (distance) of two vectors.
        Assumes they are the same length."""
    return math.sqrt(norm_squared(v1, v2))


@cuda.jit(device=True)
def calc_x_at_segment_beginning(pipe, pipe_segment):
    x_beginning = pipe[0][0]
    for i in range(0, pipe_segment):
        x_beginning += pipe[i][4]
    return x_beginning


@cuda.jit(device=True)
def is_out_of_pipe(position, pipe, pipe_segment):
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    radius_in_position = calc_radius_in_position(position, pipe, pipe_segment)
    return h > radius_in_position


@cuda.jit(device=True)
def calc_radius_in_position(position, pipe, pipe_segment):
    if pipe[pipe_segment][3] == pipe[pipe_segment + 1][3]:
        return pipe[pipe_segment][3]
    elif pipe[pipe_segment][3] < pipe[pipe_segment + 1][3]:
        delta = position[0] - calc_x_at_segment_beginning(pipe, pipe_segment)
        truncated_length = pipe[pipe_segment][4] * pipe[pipe_segment][3] / (pipe[pipe_segment + 1][3] -
                                                                            pipe[pipe_segment][3])
        return pipe[pipe_segment][3] * (1.0 + delta / truncated_length)
    else:
        delta = calc_x_at_segment_beginning(pipe, pipe_segment) + pipe[pipe_segment][4] - position[0]
        truncated_length = pipe[pipe_segment][4] * pipe[pipe_segment + 1][3] / (pipe[pipe_segment][3]
                                                                                - pipe[pipe_segment + 1][3])
        return pipe[pipe_segment + 1][3] * (1.0 + delta / truncated_length)


@cuda.jit(device=True)
def find_segment(position, pipe):
    for j in range(0, pipe.shape[0] - 1):
        if pipe[j][0] <= position[0] < pipe[j + 1][0]:
            return j
    return -1


@cuda.jit(device=True)
def calc_cos_between_vectors(first_vector, second_vector, first_vector_length, second_vector_length):
    scalar = 0
    for dim in range(3):
        scalar = scalar + first_vector[dim] * second_vector[dim]
    return scalar / (first_vector_length * second_vector_length)


@cuda.jit(device=True)
def calc_summary_vector(l_vector, p_vector, summary_vector):
    l_vector_length = calc_vector_length(l_vector)
    p_vector_length = calc_vector_length(p_vector)
    cos_value = calc_cos_between_vectors(l_vector, p_vector, l_vector_length, p_vector_length)
    for dim in range(0, 3):
        summary_vector[dim] = 2.0 * l_vector[dim] / l_vector_length * cos_value * p_vector_length


@cuda.jit(device=True)
def calc_vector_length(vector):
    length = 0
    for i in range(0, len(vector)):
        length += vector[i] ** 2
    return math.sqrt(length)


@cuda.jit(device=True)
def calc_distance_between_points(first, second):
    length = 0
    for i in range(0, len(first)):
        length += (first[i] - second[i]) ** 2
    return math.sqrt(length)


@cuda.jit(device=True)
def calc_dt(position, speed, edge_vector, l_point):
    speed_length = calc_vector_length(speed)
    edge_vector_length = calc_vector_length(edge_vector)
    cos_a = calc_cos_between_vectors(speed, edge_vector, speed_length, edge_vector_length)
    d = calc_distance_to_pipe(position, edge_vector, l_point)
    sin_a = math.sqrt(1 - cos_a**2)
    d_to_collision_point = d / (sin_a + 0.001)
    return -d_to_collision_point / speed_length


@cuda.jit(device=True)
def calc_distance_to_pipe(point, edge_vector, l_point):
    points_vector = cuda.local.array(3, np.double)
    for dim in range(3):
        points_vector[dim] = l_point[dim] - point[dim]

    vector_scalar = cuda.local.array(3, np.double)
    calc_vector_scalar(vector_scalar, edge_vector, points_vector)

    vector_scalar_length = calc_vector_length(vector_scalar)
    edge_vector_length = calc_vector_length(edge_vector)
    return vector_scalar_length / edge_vector_length


@cuda.jit(device=True)
def calc_vector_scalar(result, first, second):
    result[0] = first[1]*second[2] - first[2]*second[1]
    result[1] = first[2]*second[0] - first[0]*second[2]
    result[2] = first[0]*second[1] - first[1]*second[0]


@cuda.jit(device=True)
def calc_h(position, pipe):
    y_norm = position[1] - pipe[0][1]
    z_norm = position[2] - pipe[0][2]
    return math.sqrt(y_norm**2 + z_norm**2)


@cuda.jit(device=True)
def calc_edge_vector(edge_vector, first_l_point, position, pipe, pipe_segment):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment + 1][3]
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    second_l_point = cuda.local.array(3, np.double)
    first_l_point[0] = pipe[pipe_segment][0]
    second_l_point[0] = pipe[pipe_segment + 1][0]
    for dim in range(1, 3):
        first_l_point[dim] = (position[dim] - pipe[pipe_segment][dim]) / h * start_radius
        second_l_point[dim] = (position[dim] - pipe[pipe_segment][dim]) / h * end_radius

    for dim in range(3):
        edge_vector[dim] = second_l_point[dim] - first_l_point[dim]

    for dim in range(1, 3):
        first_l_point[dim] = first_l_point[dim] + pipe[pipe_segment][dim]


@cuda.jit(device=True)
def calc_collision_point(collision_point, position, speed, edge_vector, l_point):
    dt = calc_dt(position, speed, edge_vector, l_point)

    for dim in range(3):
        collision_point[dim] = position[dim] + speed[dim] * dt


@cuda.jit(device=True)
def solve_collision(position, speed, pipe, pipe_segment):
    collision_point = cuda.local.array(3, np.double)

    edge_vector = cuda.local.array(3, np.double)
    first_l_point = cuda.local.array(3, np.double)
    calc_edge_vector(edge_vector, first_l_point, position, pipe, pipe_segment)

    calc_collision_point(collision_point, position, speed, edge_vector, first_l_point)
    if pipe[pipe_segment][3] == pipe[pipe_segment+1][3]:
        for dim in range(1, 3):
            speed[dim] = -speed[dim]
    else:
        summary_vector = cuda.local.array(3, np.double)
        calc_summary_vector(edge_vector, speed, summary_vector)

        for dim in range(3):
            speed[dim] = summary_vector[dim] - speed[dim]

    way_after_collision = calc_distance_between_points(position, collision_point)
    speed_vec_length = calc_vector_length(speed)
    for dim in range(3):
        position[dim] = collision_point[dim] + speed[dim] * way_after_collision / speed_vec_length
