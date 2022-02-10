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
def solve_collision(position, speed, i, pipe, pipe_segment):
    start_radius = pipe[pipe_segment][3]
    end_radius = pipe[pipe_segment + 1][3]
    y_norm = position[1] - pipe[pipe_segment][1]
    z_norm = position[2] - pipe[pipe_segment][2]
    h = (y_norm ** 2 + z_norm ** 2) ** 0.5

    edge_vector = cuda.local.array(3, np.double)

    if start_radius == end_radius:
        edge_vector[0] = 1
        edge_vector[1] = 0
        edge_vector[2] = 0
    else:
        edge_vector[0] = pipe[pipe_segment][4]
        edge_vector[1] = abs(position[i][1]/h*(start_radius - end_radius))
        edge_vector[2] = abs(position[i][2]/h*(start_radius - end_radius))
        edge_vector_length = (edge_vector[0]**2+edge_vector[1]**2+edge_vector[2]**2)**0.5
        for dim in range(0, 2):
            edge_vector[dim] = edge_vector[dim] / edge_vector_length


@cuda.jit()
def collision_kernel(
    d_position: np.ndarray,
    d_velocity: np.ndarray,
    pipe: np.ndarray
):
    i = get_index()
    if i >= d_position.shape[0]:
        return

    pipe_index = find_segment(d_position[i], pipe)
    if pipe_index == -1:  # element is outside
        return

    if is_out_of_pipe(d_position[i], pipe, pipe_index):
        solve_collision(d_position, d_velocity, i, pipe, pipe_index)
