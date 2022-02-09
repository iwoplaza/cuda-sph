from vis.src.main.vector import Vec3f
from ..component import Component


class Cube(Component):
    def __init__(self, origin: Vec3f, scale: Vec3f):
        self.origin = origin
        self.scale = scale

    def draw(self, delta_time: float):
        raise NotImplementedError()
