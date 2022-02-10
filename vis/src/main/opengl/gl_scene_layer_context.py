from vis.src.main.abstract import SceneLayerContext
from vis.src.main.vector import Vec3f
from .scene_components import GLPointField, GLCube, GLCamera
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
