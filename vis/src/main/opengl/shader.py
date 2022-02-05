from OpenGL.GL import *
from OpenGL.GL import shaders


class Shader:
    def __init__(self, vert_shader_src, frag_shader_src):
        super().__init__()

        vertex_shader = shaders.compileShader(vert_shader_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(frag_shader_src, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex_shader, fragment_shader)

    def use(self):
        glUseProgram(self.program)

    def set_projection_matrix(self, mat):
        raise NotImplementedError()
