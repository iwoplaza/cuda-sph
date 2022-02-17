from typing import List

from vis.src.abstract.scene_components import PointField
from vis.src.vector import Vec3f
from ..common_shaders import CommonShaders
from OpenGL.GL import *
from OpenGL.GLUT import *
import glm
import numpy as np

MAX_PARTICLES = 100000


class GLPointField(PointField):
    def __init__(self, origin: Vec3f, scale: Vec3f):
        super().__init__(origin=origin, scale=scale)

        self.shader = CommonShaders.POINT_FIELD
        self.point_count = 0

        # Tried without a shared VBO, didn't work, so have to pass at least a float.
        vertex_attributes = np.array([0], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Shared vbo
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes, GL_STATIC_DRAW)
        float_size = sizeof(GLfloat)
        glVertexAttribPointer(0, 1, GL_FLOAT, False, 1 * float_size, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Per-instance positions vbo
        self.positions_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_vbo)
        glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 3 * float_size, None, GL_STREAM_DRAW)
        glVertexAttribPointer(1,
                              3,  # size
                              GL_FLOAT,  # type
                              False,  # normalized
                              3 * float_size,  # stride
                              None)
        glVertexAttribDivisor(1, 1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Per-instance densities vbo
        self.densities_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.densities_vbo)
        glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * float_size, None, GL_STREAM_DRAW)
        glVertexAttribPointer(2,
                              1,  # size
                              GL_FLOAT,  # type
                              False,  # normalized
                              1 * float_size,  # stride
                              None)
        glVertexAttribDivisor(2, 1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

    def set_point_positions(self, positions: np.ndarray):
        positions = np.asarray(positions, dtype=np.float32)
        arr = positions.reshape(len(positions)*3)
        self.point_count = len(positions)

        glBindBuffer(GL_ARRAY_BUFFER, self.positions_vbo)
        glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 3 * sizeof(GLfloat), None, GL_STREAM_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(arr) * sizeof(GLfloat), arr)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_point_densities(self, densities: np.ndarray):
        densities = np.asarray(densities, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.densities_vbo)
        glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(GLfloat), None, GL_STREAM_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(densities) * sizeof(GLfloat), densities)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, delta_time: float):
        self.shader.use()
        self.shader.set_model_matrix(glm.mat4(1))

        # for i, offset in enumerate(self.point_positions):
        #     glUniform3fv(10 + i, 1, glm.value_ptr(glm.vec3(*offset)))

        glBindVertexArray(self.vao)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glDrawArraysInstanced(GL_POINTS, 0, 1, self.point_count)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glBindVertexArray(0)
