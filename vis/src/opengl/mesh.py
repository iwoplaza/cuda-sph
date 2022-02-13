from typing import Tuple
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np


class Mesh:
    def __init__(self, vao):
        self.vao = vao

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)


def make_rectangle_mesh(
        top_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
        uv_min: Tuple[float, float] = (0,  0),
        uv_max: Tuple[float, float] = (1, 1),
        ):

    vertex_attributes = np.array([
        # x  y  u  v
        top_left[0],     top_left[1],     uv_min[0], uv_min[1],
        top_left[0],     bottom_right[1], uv_min[0], uv_max[1],
        bottom_right[0], bottom_right[1], uv_max[0], uv_max[1],
        top_left[0],     top_left[1],     uv_min[0], uv_min[1],
        bottom_right[0], bottom_right[1], uv_max[0], uv_max[1],
        bottom_right[0], top_left[1],     uv_max[0], uv_min[1]
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_attributes, GL_STATIC_DRAW)
    float_size = vertex_attributes.itemsize
    glVertexAttribPointer(0, 2, GL_FLOAT, False, 4 * float_size, None)
    glVertexAttribPointer(1, 2, GL_FLOAT, False, 4 * float_size, c_void_p(2 * float_size))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindVertexArray(0)

    return Mesh(vao)
