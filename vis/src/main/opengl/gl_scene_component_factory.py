from vis.src.main.abstract import SceneComponentFactory
from vis.src.main.vector import Vec3f
from .scene_components import GLPointField, GLCube
from .gl_window import GLWindow


class GLSceneComponentFactory(SceneComponentFactory):
    window: GLWindow

    def __init__(self, window: GLWindow):
        self.window = window

    def create_point_field(self, origin: Vec3f, scale: Vec3f):
        return GLPointField(origin, scale)

    def create_cube(self, origin: Vec3f, scale: Vec3f):
        return GLCube(origin, scale)
