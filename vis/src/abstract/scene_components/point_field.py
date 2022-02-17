import numpy as np
from vis.src.vector import Vec3f
from ..component import Component


class PointField(Component):
    def __init__(self, origin: Vec3f, scale: Vec3f):
        self.origin = origin
        self.scale = scale

    def set_point_positions(self, positions: np.ndarray):
        raise NotImplementedError()

    def set_point_densities(self, densities: np.ndarray):
        raise NotImplementedError()

    def draw(self, delta_time: float):
        raise NotImplementedError()
