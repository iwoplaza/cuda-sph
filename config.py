import numpy as np


# physics constants
MASS:  np.float64 = 1.0
RHO_0: np.float64 = 1.0
INF_R: np.float64 = 5
VISC:  np.float64 = 0.5
K:     np.float64 = 20.0
SCALE: np.float64 = 2e6
DAMP:  np.float64 = 0.45

# default simulation parameters
DEFAULT_DURATION = 20
DEFAULT_FPS = 30
DEFAULT_N_PARTICLES = 5_000
DEFAULT_EXT_FORCE = np.array([2.0, -2.0, 2.0], dtype=np.float64)  # (x,y,z)

DEFAULT_SPACE_SIDE_LENGTH = 20 * INF_R
DEFAULT_VOXEL_SIDE_LENGTH = INF_R
DEFAULT_SPACE_SIZE = np.array([DEFAULT_SPACE_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)
DEFAULT_VOXEL_SIZE = np.array([DEFAULT_VOXEL_SIDE_LENGTH for _ in range(3)], dtype=np.float64)  # (x,y,z)

# neighbours settings (tbh it should never be changed XD)
MAX_NEIGHBOURS = 32
NEIGHBOURING_VOXELS_COUNT = 27




