from numba import cuda
import numpy as np
import math
import numba.cuda.random as random

from config import W_CONST, INF_R_2, GRAD_W_CONST, INF_R, LAP_W_CONST, DAMP, MASS, DEFAULT_SPEED


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
def compute_w(pos_i, pos_j):
    """Computes a W kernel (SPH weight) for two positions"""
    return W_CONST * (INF_R_2 - norm_squared(pos_i, pos_j)) ** 3


@cuda.jit(device=True)
def compute_grad_w(pos_i, pos_j, result_vec):
    """Computes a gradient of W kernel (SPH weight) for two positions"""

    dist = norm(pos_i, pos_j)

    factor = GRAD_W_CONST * (INF_R - dist)**2

    for dim in range(3):
        result_vec[dim] = factor * (pos_i[dim] - pos_j[dim]) / dist


@cuda.jit(device=True)
def compute_lap_w(pos_i, pos_j):
    """Computes a laplacian of W kernel (SPH weight) for two positions"""
    return LAP_W_CONST * (INF_R - norm(pos_i, pos_j))


@cuda.jit
def integrating_kernel(
        result_force: np.ndarray,
        updated_position: np.ndarray,
        updated_velocity: np.ndarray,
        external_force: np.ndarray,
        density: np.ndarray,
        pressure_term: np.ndarray,
        viscosity_term: np.ndarray,
        DT: np.float64,
):
    i = get_index()
    if i >= updated_position.shape[0]:
        return

    # perform numerical integration with 'dt' time step (in seconds)
    for dim in range(3):
        result_force[i][dim] = (
                external_force[dim] +
                -pressure_term[i][dim] +
                viscosity_term[i][dim]
        )
        updated_velocity[i][dim] += result_force[i][dim] / density[i] * DT
        updated_position[i][dim] += updated_velocity[i][dim] * DT


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
def calc_collision_move_vector_length(l_point, l_vector, p_point, p_vector) -> np.double:
    for dim1 in range(0, 3):
        for dim2 in range(dim1+1, 3):
            denominator = p_vector[dim1] * l_vector[dim2] - p_vector[dim2] * l_vector[dim1]
            if abs(denominator) > 0.001:
                nominator = l_vector[dim1] * p_point[dim2] - l_vector[dim1] * l_point[dim2] - \
                            l_vector[dim2] * p_point[dim1] + l_vector[dim2] * l_point[dim1]
                return nominator / denominator
    # rare example where denominator is always 0
    for dim in range(0, 3):
        if l_vector[dim] == 0 and p_vector[dim] != 0:
            return (l_point[dim] - p_point[dim]) / p_vector[dim]
    return calc_vector_length(p_vector)/30


@cuda.jit(device=True)
def calc_summary_vector(l_vector, p_vector, summary_vector):
    scalar = 0
    for dim in range(3):
        scalar = scalar + l_vector[dim] * p_vector[dim]
    l_vector_length = calc_vector_length(l_vector)
    p_vector_length = calc_vector_length(p_vector)
    cos_value = scalar / (l_vector_length * p_vector_length)

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
def solve_collision(position, speed, pipe, pipe_segment):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment + 1][3]
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    edge_vector = cuda.local.array(3, np.double)

    t = -0.01
    if start_radius == end_radius:
        #t = -0.1  # don't need for solving equation

        collision_point = cuda.local.array(3, np.double)  # calculating collision point
        for dim in range(3):
            collision_point[dim] = position[dim] + speed[dim] * t

        way_after_collision = calc_distance_between_points(position, collision_point)

        for dim in range(1, 3):
            speed[dim] = -speed[dim]

        speed_vec_length = calc_vector_length(speed)

        for dim in range(1, 3):
            position[dim] = collision_point[dim] + speed[dim] * way_after_collision / speed_vec_length

    else:
        first_l_point = cuda.local.array(3, np.double)
        second_l_point = cuda.local.array(3, np.double)
        first_l_point[0] = pipe[pipe_segment][0]
        second_l_point[0] = pipe[pipe_segment + 1][0]
        for dim in range(1, 3):
            first_l_point[dim] = (position[dim] - pipe[pipe_segment][dim]) / h * start_radius
            second_l_point[dim] = (position[dim] - pipe[pipe_segment][dim]) / h * end_radius

        for dim in range(3):
            edge_vector[dim] = second_l_point[dim] - first_l_point[dim]

        #t = -0.1

        collision_point = cuda.local.array(3, np.double)
        for dim in range(3):
            collision_point[dim] = position[dim] + speed[dim] * t

        summary_vector = cuda.local.array(3, np.double)
        calc_summary_vector(edge_vector, speed, summary_vector)

        for dim in range(3):
            speed[dim] = summary_vector[dim] - speed[dim]

        way_after_collision = calc_distance_between_points(position, collision_point)

        speed_vec_length = calc_vector_length(speed)

        for dim in range(3):
            position[dim] = collision_point[dim] + speed[dim] * way_after_collision / speed_vec_length


@cuda.jit()
def collision_kernel(
        position: np.ndarray,
        velocity: np.ndarray,
        pipe: np.ndarray,
        rng_states
):
    i = get_index()
    if i >= position.shape[0]:
        return

    pipe_index = find_segment(position[i], pipe)
    if pipe_index == -1:  # element is outside
        put_particle_at_pipe_begin(position[i], velocity[i], pipe, rng_states, i)
    else:
        if is_out_of_pipe(position[i], pipe, pipe_index):
            solve_collision(position[i], velocity[i], pipe, pipe_index)


@cuda.jit()
def collision_kernel_box(
        position: np.ndarray,
        velocity: np.ndarray,
        space_size: np.ndarray
):
    """
    Colliding particles with the box, which has the space size. All pipe concept is ignored.
    """
    i = get_index()
    if i >= position.shape[0]:
        return

    for dim in range(3):  # tests against all 3 dimensions
        bounced = False
        if position[i][dim] < 0:  # check if it's colliding
            position[i][dim] = 1e-3  # statically move back inside
            bounced = True
        if position[i][dim] > space_size[dim]:  # same thing to the other side
            position[i][dim] = space_size[dim] - 1e-3
            bounced = True
        if bounced:
            velocity[i][dim] *= -1  # flip velocity, to change the direction of movement
            velocity[i][dim] *= DAMP  # add a little damping, to slow the particle down after collision


@cuda.jit(device=True)
def rand_position_inside_pipe(position, pipe, rng_states, i):
    """
    Rand position inside pipe, use actual position_x point - so it should be rounded previously if need to be change
    """
    pipe_segment = find_segment(position, pipe)
    R = calc_radius_in_position(position, pipe, pipe_segment)
    r = R * math.sqrt(random.xoroshiro128p_uniform_float64(rng_states, i))
    theta = random.xoroshiro128p_uniform_float64(rng_states, i) * 2.0 * math.pi

    position[1] = pipe[0][1] + r * math.cos(theta)
    position[2] = pipe[0][2] + r * math.sin(theta)


@cuda.jit(device=True)
def put_particle_at_pipe_begin(position, velocity, pipe, rng_states, i):
    """
    Put particle at the beginning in the pipe, position y and z points are set randomly,
    speed is set to [DEFAULT_SPEED, 0, 0]
    """
    position[0] = 0.0
    rand_position_inside_pipe(position, pipe, rng_states, i)

    velocity[0] = DEFAULT_SPEED
    for dim in range(1, 3):
        velocity[dim] = 0


@cuda.jit()
def spawn_particles_inside_pipe_kernel(
        position: np.ndarray,
        pipe: np.ndarray,
        rng_states: np.ndarray
):
    """Sets all positions inside the pipe, and horizontal x direction
        "used to initialize first state of simulation"""

    i = get_index()
    if i >= position.shape[0]:
        return

    rand_position_inside_pipe(position[i], pipe, rng_states, i)
