import glm
from OpenGL.GL import *
from OpenGL.GLUT import *

from vis.src.main.abstract import Window
from .common_shaders import CommonShaders
from .shader import Shader


class GLWindow(Window):
    __ui_shaders: list[Shader]
    __scene_shaders: list[Shader]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initializing the GLUT window
        glutInit()  # Initialize a glut instance which will allow us to customize our window
        glutInitDisplayMode(GLUT_RGBA)  # Set the display mode to be colored
        glutInitWindowSize(self.width, self.height)  # Set the width and height of your window
        glutInitWindowPosition(0, 0)  # Set the position at which this windows should appear
        wind = glutCreateWindow(self.title)  # Give your window a title

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.last_time = glutGet(GLUT_ELAPSED_TIME)
        self.__ui_shaders = CommonShaders.register_ui_shaders()
        self.__scene_shaders = CommonShaders.register_scene_shaders()
        self.ui_proj_mat = None
        self.scene_proj_mat = None

        self.__setup_projection()

    def __setup_projection(self):
        glViewport(0, 0, self.width, self.height)
        self.ui_proj_mat = glm.ortho(0, self.width, self.height, 0, -1, 1)
        self.scene_proj_mat = glm.perspective(90.0, self.width / self.height, 0.01, 1000)

        for shader in self.__ui_shaders:
            self.__update_ui_shader_uniforms(shader)

        for shader in self.__scene_shaders:
            self.__update_scene_shader_uniforms(shader)

    def __on_resize(self, w, h):
        self.width = w
        self.height = h

        self.__setup_projection()

    def __on_mouse_move(self, x: int, y: int):
        pass

    def __update_ui_shader_uniforms(self, shader):
        shader.use()
        shader.set_projection_matrix(self.ui_proj_mat)

    def __update_scene_shader_uniforms(self, shader):
        shader.use()
        shader.set_projection_matrix(self.scene_proj_mat)

    def __display_func(self):
        current_time = glutGet(GLUT_ELAPSED_TIME)
        delta = float(current_time - self.last_time) * 0.001
        self.last_time = current_time

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for layer in self.layers:
            layer.draw(delta)

        glutSwapBuffers()

    def run(self):
        glutDisplayFunc(self.__display_func)  # Tell OpenGL to call the showScreen method continuously
        glutIdleFunc(self.__display_func)  # Draw any graphics or shapes in the showScreen function at all times
        glutReshapeFunc(self.__on_resize)
        glutMotionFunc(self.__on_mouse_move)
        glutPassiveMotionFunc(self.__on_mouse_move)
        glutMainLoop()  # Keeps the window created above displaying/running in a loop
