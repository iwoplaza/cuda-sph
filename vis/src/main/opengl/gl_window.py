from typing import List

import glm
from OpenGL.GL import *
from OpenGL.GLUT import *

from vis.src.main.abstract import Window
from .common_shaders import CommonShaders
from .shader import Shader, SceneShader
from .scene_components import GLCamera


class GLWindow(Window):
    __ui_shaders: List[Shader]
    __scene_shaders: List[SceneShader]

    current_camera: GLCamera = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initializing the GLUT window
        glutInit()  # Initialize a glut instance which will allow us to customize our window
        glutInitDisplayMode(GLUT_RGBA)  # Set the display mode to be colored
        glutInitWindowSize(self.width, self.height)  # Set the width and height of your window
        glutInitWindowPosition(0, 0)  # Set the position at which this windows should appear
        wind = glutCreateWindow(self.title)  # Give your window a title

        glEnable(GL_BLEND)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.last_time = glutGet(GLUT_ELAPSED_TIME)
        self.__ui_shaders = CommonShaders.register_ui_shaders()
        self.__scene_shaders = CommonShaders.register_scene_shaders()
        self.ui_proj_mat = None

        self.__setup_projection()

    def __setup_projection(self):
        glViewport(0, 0, self.width, self.height)
        self.ui_proj_mat = glm.ortho(0, self.width, self.height, 0, -1, 1)

        for shader in self.__ui_shaders:
            self.__update_ui_shader_uniforms(shader)

        if self.current_camera is not None:
            self.current_camera.setup_projection(self.width, self.height)

            for shader in self.__scene_shaders:
                self.__update_scene_shader_uniforms(shader)

    def __on_resize(self, w, h):
        self.width = w
        self.height = h

        self.__setup_projection()

    def __on_mouse_move(self, x: int, y: int):
        for layer in reversed(self.layers):
            layer.on_mouse_move(x, y)

    def __on_mouse_action(self, button: int, state: int,
                          x: int, y: int):
        if state == GLUT_DOWN:
            for layer in reversed(self.layers):
                if layer.on_mouse_btn_pressed(x, y, button):
                    break
        else:
            for layer in self.layers:
                layer.on_mouse_btn_released(x, y, button)

    def __on_key_press(self, key, x, y):
        for layer in reversed(self.layers):
            if layer.on_key_pressed(key):
                break

    def __on_key_release(self, key, x, y):
        for layer in self.layers:
            layer.on_key_released(key)

    def __update_ui_shader_uniforms(self, shader):
        shader.use()
        shader.set_projection_matrix(self.ui_proj_mat)

    def __update_scene_shader_uniforms(self, shader):
        shader.use()
        shader.set_projection_matrix(self.current_camera.proj_mat)

    def __display_func(self):
        current_time = glutGet(GLUT_ELAPSED_TIME)
        delta = float(current_time - self.last_time) * 0.001
        self.last_time = current_time

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.current_camera is not None:
            self.current_camera.update(delta)

            for shader in self.__scene_shaders:
                shader.use()
                shader.set_view_matrix(self.current_camera.view_mat)

        for layer in self.layers:
            layer.draw(delta)

        glutSwapBuffers()

    def use_camera(self, camera):
        self.current_camera = camera

    def perform_command(self, command):
        if command.type == 'position-camera':
            self.current_camera.set_position(command.position)
            if command.yaw is not None:
                self.current_camera.set_yaw(command.yaw)
            if command.pitch is not None:
                self.current_camera.set_pitch(command.pitch)
        else:
            print(f"Tried to perform unknown command: \"{command.type}\". Details: {dir(command)}")

    def run(self):
        glutDisplayFunc(self.__display_func)  # Tell OpenGL to call the showScreen method continuously
        glutIdleFunc(self.__display_func)  # Draw any graphics or shapes in the showScreen function at all times
        glutReshapeFunc(self.__on_resize)
        glutMotionFunc(self.__on_mouse_move)
        glutPassiveMotionFunc(self.__on_mouse_move)
        glutMouseFunc(self.__on_mouse_action)
        glutKeyboardFunc(self.__on_key_press)
        glutKeyboardUpFunc(self.__on_key_release)
        glutMainLoop()  # Keeps the window created above displaying/running in a loop
