from vis.src.vector import Vec3f
from ..component import Component


class Camera(Component):
    def __init__(self, origin: Vec3f):
        pass

    def enable(self):
        raise NotImplementedError()

    def set_position(self, position: Vec3f):
        raise NotImplementedError()

    def set_yaw(self, yaw: float):
        raise NotImplementedError()

    def set_pitch(self, pitch: float):
        raise NotImplementedError()
