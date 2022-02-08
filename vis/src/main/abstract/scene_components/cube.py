from vis.src.main.vector import Vec3f


class Cube:
    def __init__(self, origin: Vec3f, scale: Vec3f):
        self.origin = origin
        self.scale = scale

    def draw(self, deltaTime: float):
        raise NotImplementedError()
