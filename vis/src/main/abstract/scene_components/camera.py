from vis.src.main.vector import Vec3f
from ..component import Component


class Camera(Component):
    def __init__(self, origin: Vec3f):
        pass

    def enable(self):
        raise NotImplementedError()
