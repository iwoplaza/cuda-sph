from common.data_classes import Pipe
from vis.src.abstract.scene_components import PointField, Cube, Camera, WireCylinder
from vis.src.vector import Vec3f


class SceneComponentFactory:
    def create_point_field(self, origin: Vec3f, scale: Vec3f) -> PointField:
        raise NotImplementedError()

    def create_cube(self, origin: Vec3f, scale: Vec3f) -> Cube:
        raise NotImplementedError()

    def create_camera(self, origin: Vec3f, yaw: float = None, pitch: float = None) -> Camera:
        raise NotImplementedError()

    def create_wire_cylinder(self, start: Vec3f, end: Vec3f, start_radius: float, end_radius: float) -> WireCylinder:
        raise NotImplementedError()

    def create_wire_cube(self, position: Vec3f, scale: Vec3f):
        raise NotImplementedError()

    def create_wire_pipe(self, pipe: Pipe):
        raise NotImplementedError()

