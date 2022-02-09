from typing import List

from vis.src.main.vector import Vec3f


class PointField:
    def __init__(self, origin: Vec3f, scale: Vec3f):
        self.origin = origin
        self.scale = scale

    def set_point_positions(self, positions: List[Vec3f]):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()
