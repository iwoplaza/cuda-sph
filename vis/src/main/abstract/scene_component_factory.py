from vis.src.main.abstract.scene_components import PointField, Cube
from vis.src.main.vector import Vec3f


class SceneComponentFactory:
    def create_point_field(self, origin: Vec3f, scale: Vec3f) -> PointField:
        raise NotImplementedError()

    def create_cube(self, origin: Vec3f, scale: Vec3f) -> Cube:
        raise NotImplementedError()
