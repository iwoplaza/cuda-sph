import os
from typing import Dict, Tuple
from OpenGL.GL import *
from OpenGL.GL import shaders
import glm

import config


def dict_get_or(d, key: str, default):
    return d[key] if key in d else default


class Shader:
    def __init__(self, vert_shader_src, frag_shader_src, uniform_indices: Dict[str, int] = None):
        super().__init__()

        if uniform_indices is None:
            self.uniform_indices = dict()
        else:
            self.uniform_indices = uniform_indices

        self.proj_matrix_uniform_id = dict_get_or(self.uniform_indices, 'projection_matrix', default=0)
        self.model_matrix_uniform_id = dict_get_or(self.uniform_indices, 'model_matrix', default=1)

        vertex_shader = shaders.compileShader(vert_shader_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(frag_shader_src, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex_shader, fragment_shader)

    def use(self):
        glUseProgram(self.program)

    def set_projection_matrix(self, mat):
        glUniformMatrix4fv(self.proj_matrix_uniform_id, 1, GL_FALSE, glm.value_ptr(mat))

    def set_model_matrix(self, mat):
        glUniformMatrix4fv(self.model_matrix_uniform_id, 1, GL_FALSE, glm.value_ptr(mat))


class SceneShader(Shader):
    def __init__(self, vert_shader_src, frag_shader_src, uniform_indices: Dict[str, int] = None):
        super().__init__(vert_shader_src, frag_shader_src, uniform_indices)

        self.view_matrix_uniform_id = dict_get_or(self.uniform_indices, 'view_matrix', default=2)

    def set_view_matrix(self, mat):
        glUniformMatrix4fv(self.view_matrix_uniform_id, 1, GL_FALSE, glm.value_ptr(mat))


class SolidShader(SceneShader):
    def __init__(self, vert_shader_src, frag_shader_src, uniform_indices: Dict[str, int] = None):
        super().__init__(vert_shader_src, frag_shader_src, uniform_indices)

        self.color_uniform_id = glGetUniformLocation(self.program, "uColor")

    def set_color(self, col: Tuple[float, float, float, float]):
        glUniform4fv(self.color_uniform_id, 1, col)


class UISolidShader(Shader):
    def __init__(self, vert_shader_src, frag_shader_src, uniform_indices: Dict[str, int] = None):
        super().__init__(vert_shader_src, frag_shader_src, uniform_indices)

        self.color_uniform_id = glGetUniformLocation(self.program, "uColor")

    def set_color(self, col: Tuple[float, float, float, float]):
        glUniform4fv(self.color_uniform_id, 1, col)


def load_shader_src_from_asset(vert_shader_name, frag_shader_name=None):
    if frag_shader_name is None:
        frag_shader_name = vert_shader_name

    with open(os.path.join(config.ROOT_PROJ_DIRNAME, config.ASSETS_DIRNAME, f'{vert_shader_name}.vert')) as f:
        vert_shader_src = f.read()

    with open(os.path.join(config.ROOT_PROJ_DIRNAME, config.ASSETS_DIRNAME, f'{frag_shader_name}.frag')) as f:
        frag_shader_src = f.read()

    return {'vert_shader_src': vert_shader_src, 'frag_shader_src': frag_shader_src}
