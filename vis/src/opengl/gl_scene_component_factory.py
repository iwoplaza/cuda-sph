from common.data_classes import Pipe
from vis.src.abstract import SceneComponentFactory
from vis.src.vector import Vec3f
from .scene_components import GLPointField, GLCube, GLCamera, GLWireCylinder
from .gl_window import GLWindow
from .scene_components.gl_wire_cube import GLWireCube
from .scene_components.gl_wire_pipe import GLWirePipe


class GLSceneComponentFactory(SceneComponentFactory):
    __window: GLWindow

    def __init__(self, window: GLWindow):
        self.__window = window

    def create_point_field(self, origin: Vec3f, scale: Vec3f):
        return GLPointField(origin, scale)

    def create_cube(self, origin: Vec3f, scale: Vec3f):
        return GLCube(origin, scale)

    def create_camera(self, origin: Vec3f, yaw: float = None, pitch: float = None) -> GLCamera:
        return GLCamera(self.__window, origin, yaw = yaw, pitch = pitch)

    def create_wire_cylinder(self, start: Vec3f, end: Vec3f, start_radius: float, end_radius: float) -> GLWireCylinder:
        return GLWireCylinder(start, end, start_radius, end_radius)

    def create_wire_cube(self, position: Vec3f, scale: Vec3f):
        return GLWireCube(position, scale)

    def create_wire_pipe(self, pipe: Pipe):
        return GLWirePipe(pipe)
