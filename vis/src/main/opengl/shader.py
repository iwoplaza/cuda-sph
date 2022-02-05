from OpenGL.GL import *
from OpenGL.GL import shaders
import glm


class Shader:
    def __init__(self, vert_shader_src, frag_shader_src, proj_matrix_uniform_id=1):
        super().__init__()

        self.proj_matrix_uniform_id = proj_matrix_uniform_id

        vertex_shader = shaders.compileShader(vert_shader_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(frag_shader_src, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex_shader, fragment_shader)

    def use(self):
        glUseProgram(self.program)

    def set_projection_matrix(self, mat):
        glUniformMatrix4fv(self.proj_matrix_uniform_id, 1, GL_FALSE, glm.value_ptr(mat))


class FileShader(Shader):
    def __init__(self, vert_shader_path, frag_shader_path, proj_matrix_uniform_id=1):
        with open(f'assets/{vert_shader_path}.vert') as f:
            text_shader_vert = f.read()

        with open(f'assets/{frag_shader_path}.frag') as f:
            text_shader_frag = f.read()

        super().__init__(text_shader_vert, text_shader_frag, proj_matrix_uniform_id)
