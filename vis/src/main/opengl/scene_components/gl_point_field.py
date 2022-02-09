from vis.src.main.abstract.scene_components import PointField
from vis.src.main.vector import Vec3f
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np


class GLPointField(PointField):
    def __init__(self, origin: Vec3f, scale: Vec3f):
        super().__init__(origin=origin, scale=scale)

        self.point_positions = []

        vertex_attributes = np.array([
            # x y
            0, 0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes, GL_STATIC_DRAW)
        float_size = vertex_attributes.itemsize
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 2 * float_size, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBindVertexArray(0)

    def set_point_positions(self, positions: list[Vec3f]):
        self.point_positions = positions

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_POINTS, 0, 1, len(self.point_positions))
        glBindVertexArray(0)
