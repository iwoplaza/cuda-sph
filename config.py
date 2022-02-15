from math import pi as PI
import numpy as np


# physics constants
MASS:  np.float64 = 0.001
RHO_0: np.float64 = 1.0
INF_R: np.float64 = 2.0
VISC:  np.float64 = 0.5
K:     np.float64 = 10.0
DAMP:  np.float64 = 0.7
INF_R_2 = INF_R ** 2
INF_R_6 = INF_R ** 6
INF_R_9 = INF_R ** 9
W_CONST = 315.0 / (64.0 * PI * INF_R_9)
GRAD_W_CONST = -45.0 / (PI * INF_R_6)
LAP_W_CONST = 45.0 / (PI * INF_R_6)


# default simulation parameters
DEFAULT_DURATION = 20
DEFAULT_FPS = 30
DEFAULT_N_PARTICLES = 10000
DEFAULT_EXT_FORCE = np.array([1.0, -0.5, 0.0], dtype=np.float64)  # (x,y,z)
DEFAULT_SPEED = 5

DEFAULT_SPACE_SIDE_LENGTH = 20 * INF_R
DEFAULT_VOXEL_SIDE_LENGTH = INF_R
DEFAULT_SPACE_SIZE = np.array([DEFAULT_SPACE_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)
DEFAULT_VOXEL_SIZE = np.array([DEFAULT_VOXEL_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)

LONG_SPACE_SIZE = np.array(
    [20 * DEFAULT_VOXEL_SIDE_LENGTH, 3 * DEFAULT_VOXEL_SIDE_LENGTH, 3 * DEFAULT_VOXEL_SIDE_LENGTH],
    dtype=np.float64
)

# neighbours settings (tbh it should never be changed XD)
MAX_NEIGHBOURS = 32
NEIGHBOURING_VOXELS_COUNT = 27




