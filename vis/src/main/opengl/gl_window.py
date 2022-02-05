import glm
from OpenGL.GL import *
from OpenGL.GLUT import *
from src.main.abstract import Window
from .shader import Shader


class GLWindow(Window):
    __shaders: list[Shader]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initializing the GLUT window
        glutInit()  # Initialize a glut instance which will allow us to customize our window
        glutInitDisplayMode(GLUT_RGBA)  # Set the display mode to be colored
        glutInitWindowSize(self.width, self.height)  # Set the width and height of your window
        glutInitWindowPosition(0, 0)  # Set the position at which this windows should appear
        wind = glutCreateWindow(self.title)  # Give your window a title

        self.__shaders = []
        self.proj_mat = None

        self.__setup_projection()

    def register_shader(self, shader: Shader):
        self.__shaders.append(shader)
        self.__update_shader_uniforms(shader)

    def __setup_projection(self):
        self.proj_mat = glm.ortho(0, self.width, self.height, 0, -1, 1)

        for shader in self.__shaders:
            self.__update_shader_uniforms(shader)

    def __update_shader_uniforms(self, shader):
        shader.use()
        shader.set_projection_matrix(self.proj_mat)

    def __display_func(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        super().draw_current_screen()

        # proj = glm.ortho(0, 500, 500, 0, -1, 1)
        # glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(proj))

        glutSwapBuffers()

    def run(self):
        glutDisplayFunc(self.__display_func)  # Tell OpenGL to call the showScreen method continuously
        glutIdleFunc(self.__display_func)  # Draw any graphics or shapes in the showScreen function at all times
        glutMainLoop()  # Keeps the window created above displaying/running in a loop
