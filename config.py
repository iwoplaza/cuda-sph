from math import pi as PI
import numpy as np


# physics constants
MASS:  np.float64 = 1e2
RHO_0: np.float64 = 1.0
INF_R: np.float64 = 5.0
VISC:  np.float64 = 0.5
K:     np.float64 = 20.0
DAMP:  np.float64 = 0.45
INF_R_2 = INF_R ** 2
INF_R_6 = INF_R ** 6
INF_R_9 = INF_R ** 9
W_CONST = 315.0 / (64.0 * PI * INF_R_9)
GRAD_W_CONST = -45.0 / (PI * INF_R_6)
LAP_W_CONST = 45.0 / (PI * INF_R_6)


# default simulation parameters
DEFAULT_DURATION = 10
DEFAULT_FPS = 1
DEFAULT_N_PARTICLES = 3_000
DEFAULT_EXT_FORCE = np.array([0.0, -100.0, 0.0], dtype=np.float64)  # (x,y,z)

DEFAULT_SPACE_SIDE_LENGTH = 20 * INF_R
DEFAULT_VOXEL_SIDE_LENGTH = INF_R
DEFAULT_SPACE_SIZE = np.array([DEFAULT_SPACE_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)
DEFAULT_VOXEL_SIZE = np.array([DEFAULT_VOXEL_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)

# neighbours settings (tbh it should never be changed XD)
MAX_NEIGHBOURS = 32
NEIGHBOURING_VOXELS_COUNT = 27




