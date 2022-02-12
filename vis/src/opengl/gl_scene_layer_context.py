from vis.src.abstract import SceneLayerContext
from vis.src.vector import Vec3f
from .scene_components import GLPointField, GLCube, GLCamera, GLWireCylinder
from .gl_window import GLWindow


class GLSceneLayerContext(SceneLayerContext):
    __window: GLWindow

    def __init__(self, window: GLWindow):
        self.__window = window

    def dispatch_command(self, command) -> None:
        self.__window.perform_command(command)

    def create_point_field(self, origin: Vec3f, scale: Vec3f):
        return GLPointField(origin, scale)

    def create_cube(self, origin: Vec3f, scale: Vec3f):
        return GLCube(origin, scale)

    def create_camera(self, origin: Vec3f, yaw: float = None, pitch: float = None) -> GLCamera:
        return GLCamera(self.__window, origin, yaw = yaw, pitch = pitch)

    def create_wire_cylinder(self, start: Vec3f, end: Vec3f, start_radius: float, end_radius: float) -> GLWireCylinder:
        return GLWireCylinder(start, end, start_radius, end_radius)
