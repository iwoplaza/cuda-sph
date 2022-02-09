from vis.src.main.abstract.scene_components import PointField, Cube, Camera
from vis.src.main.vector import Vec3f


class SceneLayerContext:
    def dispatch_command(self, command) -> None:
        raise NotImplementedError()

    def create_point_field(self, origin: Vec3f, scale: Vec3f) -> PointField:
        raise NotImplementedError()

    def create_cube(self, origin: Vec3f, scale: Vec3f) -> Cube:
        raise NotImplementedError()

    def create_camera(self, origin: Vec3f) -> Camera:
        raise NotImplementedError()
