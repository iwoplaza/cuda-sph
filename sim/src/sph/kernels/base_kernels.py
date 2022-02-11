from numba import cuda
import numpy as np
import math


@cuda.jit(device=True)
def get_index():
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    return block_width * block_idx + th_idx


@cuda.jit
def integrating_kernel(
        updated_position: np.ndarray,
        updated_velocity: np.ndarray,
        external_force: np.ndarray,
        pressure_term: np.ndarray,
        viscosity_term: np.ndarray,
        DT: np.float64,
        MASS: np.float64
):
    i = get_index()
    if i >= updated_position.shape[0]:
        return

    # perform numerical integration with 'dt' timestep (in seconds)
    result_force = cuda.local.array(3, np.float64)
    for dim in range(3):
        result_force[dim] = (
                external_force[dim] +
                pressure_term[i][dim] +
                viscosity_term[i][dim]
        )
        updated_velocity[i][dim] += result_force[dim] / MASS * DT
        updated_position[i][dim] += updated_velocity[i][dim] * DT


@cuda.jit(device=True)
def calc_x_at_segment_beginning(pipe, pipe_segment):
    x_beginning = pipe[0][0]
    for i in range(0, pipe_segment):
        x_beginning += pipe[i][4]
    return x_beginning


@cuda.jit(device=True)
def is_out_of_pipe(position, pipe, pipe_segment):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment+1][3]
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    if start_radius == end_radius:
        return h > start_radius
    elif start_radius < end_radius:
        delta = position[0] - calc_x_at_segment_beginning(pipe, pipe_segment)
        truncated_length = pipe[pipe_segment][4]*start_radius/(end_radius-start_radius)
        return h > start_radius*(1.0 + delta/truncated_length)
    else:
        delta = calc_x_at_segment_beginning(pipe, pipe_segment) + pipe[pipe_segment][4] - position[0]
        truncated_length = pipe[pipe_segment][4]*end_radius/(start_radius-end_radius)
        return h > end_radius*(1.0 + delta/truncated_length)


@cuda.jit(device=True)
def find_segment(position, pipe):
    for j in range(0, pipe.shape[0]-1):
        if pipe[j][0] <= position[0] < pipe[j+1][0]:
            return j
    return -1


@cuda.jit(device=True)
def calc_collision_move_vector_length(l_point, l_vector, p_point, p_vector) -> np.double:
    for dim1 in range(0, 3):
        for dim2 in range(dim1, 3):
            denominator = p_vector[dim1]*l_vector[dim2] - p_vector[dim2]*l_vector[dim1]
            if denominator != 0:
                nominator = l_vector[dim1]*p_point[dim2] - l_vector[dim1]*l_point[dim2] -\
                            l_vector[dim2]*p_point[dim1] + l_vector[dim2]*l_point[dim1]
                return nominator/denominator
    # rare example where nominator is always 0
    for dim in range(0, 3):
        if l_vector[dim] == 0 and p_vector[dim] != 0:
            return (l_point[dim]-p_point) / p_vector[dim]
    print("Cannot cals collision point")


@cuda.jit(device=True)
def calc_summary_vector(l_vector, p_vector):
    scalar = 0
    for dim in range(0, 3):
        scalar = scalar + l_vector[dim]*p_vector[dim]
    l_vector_length = (l_vector[0]**2+l_vector[1]**2+l_vector[2]**2)**0.5
    p_vector_length = (l_vector[0] ** 2 + l_vector[1] ** 2 + l_vector[2] ** 2) ** 0.5
    cos_value = scalar / (l_vector_length + p_vector_length)

    summary_vector = cuda.local.array(3, np.double)
    for dim in range(0, 3):
        summary_vector[dim] = l_vector[dim]/l_vector_length*cos_value
    return summary_vector


@cuda.jit(device=True)
def calc_vector_length(vector):
    length = 0
    for i in range(0, len(vector)):
        length += vector[i]**2
    return math.sqrt(length)



@cuda.jit(device=True)
def calc_distance_between_points(first, second):
    length = 0
    for i in range(0, len(first)):
        length += (first[i]-second[i])**2
    return math.sqrt(length)


@cuda.jit(device=True)
def solve_collision(position, speed, i, pipe, pipe_segment):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment + 1][3]
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    edge_vector = cuda.local.array(3, np.double)

    if start_radius == end_radius:
        t = (position[1]/h*start_radius-position[1])/speed[1]  # don't need for solving equation

        collision_point = cuda.local.array(3, np.double)  # calculating collision point
        for dim in range(0, 3):
            collision_point[dim] = position[dim] + speed[i][dim]*t

        way_after_collision = calc_distance_between_points(position, collision_point, 3)

        for dim in range(1, 3):
            speed[dim] = -speed[dim]

        speed_vec_length = calc_vector_length(speed, 3)

        for dim in range(1, 3):
            position[dim] = collision_point[dim] + speed[dim]*way_after_collision/speed_vec_length

    else:
        first_l_point = cuda.local.array(3, np.double)  # punkty na prostej i jej wektor
        second_l_point = cuda.local.array(3, np.double)
        first_l_point[0] = pipe[pipe_segment][0] + start_radius
        second_l_point[0] = pipe[pipe_segment][0] + end_radius
        for dim in range(1, 3):
            first_l_point[dim] = (position[dim] - pipe[pipe_segment][dim])/h*start_radius
            second_l_point[dim] = (position[dim] - pipe[pipe_segment][dim])/h*end_radius

        for dim in range(0, 3):
            edge_vector[dim] = second_l_point[dim]-first_l_point[dim]

        t = calc_collision_move_vector_length(first_l_point, edge_vector, position, speed)

        collision_point = cuda.local.array(3, np.double)
        for dim in range(0, 3):
            collision_point = position[i][dim] - speed[i][dim]*t

        sum_vector = calc_summary_vector()
        for dim in range(0, 3):
            speed[i][dim] = sum_vector - speed[i][dim]

        way_after_collision = 0
        for dim in range(0, 3):
            way_after_collision = way_after_collision + (position[dim] - collision_point[dim])**2
        way_after_collision = way_after_collision**0.5

        speed_vec_length = 0
        for dim in range(1, 3):
            speed_vec_length = speed_vec_length+speed[dim]**2
        speed_vec_length = speed_vec_length**0.5

        for dim in range(1, 3):
            position[dim] = collision_point[dim] + speed[dim]*way_after_collision/speed_vec_length


@cuda.jit()
def collision_kernel(
    position: np.ndarray,
    velocity: np.ndarray,
    pipe: np.ndarray
):
    i = get_index()
    if i >= position.shape[0]:
        return

    pipe_index = find_segment(position[i], pipe)
    if pipe_index == -1:  # element is outside
        return

    if is_out_of_pipe(position[i], pipe, pipe_index):
        solve_collision(position[i], velocity[i], pipe, pipe_index)


@cuda.jit()
def collision_kernel_box(
    position:       np.ndarray,
    velocity:       np.ndarray,
    space_size:     np.ndarray
):
    """
    Colliding particles with the box, which has the space size. All pipe concept is ignored.
    """
    i = get_index()
    if i >= position.shape[0]:
        return

    for dim in range(3):            # tests against all 3 dimensions
        if position[i][dim] < 0:    # check if it's colliding
            position[i][dim] = 0    # statically move back inside
            velocity[i][dim] *= -1  # flip velocity, to change the direction of movement
        if position[i][dim] > space_size[dim]:  # same thing to the other side
            position[i][dim] = space_size[dim]
            velocity[i][dim] *= -1
