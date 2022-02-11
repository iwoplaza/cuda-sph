from numba import cuda
import numpy as np


@cuda.jit(device=True)
def get_index():
    th_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    return block_width * block_idx + th_idx


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
        x_norm = position[1] - pipe[pipe_segment][0]
        truncated_length = pipe[pipe_segment][4]*start_radius/(end_radius-start_radius)
        return h > start_radius + x_norm/truncated_length
    else:
        x_norm = pipe[pipe_segment][0] - position[1]
        truncated_length = pipe[pipe_segment][4]*end_radius/(start_radius-end_radius)
        return h > start_radius + x_norm/truncated_length


@cuda.jit(device=True)
def find_segment(position, pipe):
    for j in range(1, pipe.shape[0]-1):
        if pipe[j][0] < position[0] < pipe[j+1][0]:
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
def solve_collision(position, speed, i, pipe, pipe_segment, DT=0.01):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment + 1][3]
    y_norm = position[i][1] - pipe[pipe_segment][1]
    z_norm = position[i][2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    edge_vector = cuda.local.array(3, np.double)

    if start_radius == end_radius:
        delta_point = (position[i][1]/h*start_radius-position[i][1])/speed[i][1]  # don't need for solving equation

        collision_point = cuda.local.array(3, np.double)  # calculating collision point
        for dim in range(0, 3):
            collision_point[dim] = position[i][dim] + speed[i][dim]*delta_point

        way_after_collision = 0
        for dim in range(0, 3):
            way_after_collision = way_after_collision + (position[i][dim] - collision_point[dim])**2
        way_after_collision = way_after_collision**0.5

        for dim in range(1, 3):
            speed[i][dim] = -speed[i][dim]

        speed_vec_length = 0
        for dim in range(1, 3):
            speed_vec_length = speed_vec_length+speed[i][dim]**2
        speed_vec_length = speed_vec_length**0.5

        for dim in range(1, 3):
            position[i][dim] = collision_point[dim] + speed[i][dim]*way_after_collision/speed_vec_length

    else:
        first_l_point = cuda.local.array(3, np.double)  # punkty na prostej i jej wektor
        first_l_point[0] = pipe[pipe_segment][0] + start_radius
        first_l_point[1] = (position[i][1] - pipe[pipe_segment][1])/h*start_radius
        first_l_point[2] = (position[i][2] - pipe[pipe_segment][2])/h*start_radius
        second_l_point = cuda.local.array(3, np.double)
        second_l_point[0] = pipe[pipe_segment][0] + end_radius
        second_l_point[1] = (position[i][1] - pipe[pipe_segment][1])/h*end_radius
        second_l_point[2] = (position[i][2] - pipe[pipe_segment][2])/h*end_radius

        for dim in range(0, 3):
            edge_vector[dim] = second_l_point[dim]-first_l_point[dim]

        t = calc_collision_move_vector_length(first_l_point, edge_vector, position[i], speed[i])

        collision_point = cuda.local.array(3, np.double)
        for dim in range(0, 3):
            collision_point = position[i][dim] - speed[i][dim]*t


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
        solve_collision(position, velocity, i, pipe, pipe_index)


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

    for dim in range(3):            # test against all 3 dimensions
        if position[i][dim] < 0:    # check if it's colliding
            position[i][dim] = 0    # statically move back inside
            velocity[i][dim] *= -1  # flip velocity, to change the direction of movement
        if position[i][dim] > space_size[dim]:  # same thing to the other side
            position[i][dim] = space_size[dim]
            velocity[i][dim] *= -1
