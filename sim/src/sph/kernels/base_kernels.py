import numba.cuda.random as random
from config import W_CONST, INF_R_2, GRAD_W_CONST, INF_R, LAP_W_CONST, DAMP
from sim.src.sph.kernels.util_kernels import *


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
        if position[i][dim] < INF_R:  # check if it's colliding
            position[i][dim] = INF_R  # statically move back inside
            bounced = True
        if position[i][dim] > space_size[dim] - INF_R:  # same thing to the other side
            position[i][dim] = space_size[dim] - INF_R
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
    Put particle at the beginning in the pipe, position y and z points are set randomly.
    """
    if position[0] < 0:
        position[0] = -position[0]
        velocity[0] = -velocity[0]
        if is_out_of_pipe(position, pipe, 0):
            solve_collision(position, velocity, pipe, 0)
    else:
        position[0] = 0.0
        rand_position_inside_pipe(position, pipe, rng_states, i)


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
