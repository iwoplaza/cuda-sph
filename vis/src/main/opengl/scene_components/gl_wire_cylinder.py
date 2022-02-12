import math
import glm
import numpy as np
from OpenGL.GL import *

from vis.src.main.abstract.scene_components import WireCylinder
from vis.src.main.vector import Vec3f
from vis.src.main.opengl.common_shaders import CommonShaders


class GLWireCylinder(WireCylinder):
    def __init__(self, start: Vec3f, end: Vec3f, start_radius: float, end_radius: float):
        circular_segments = 10
        super().__init__()

        self.shader = CommonShaders.SOLID

        direction = glm.vec3(*end) - glm.vec3(*start)
        angle = math.atan2(direction.z, direction.x) - np.pi / 2

        rotation = glm.rotate(glm.mat4(1), angle, glm.vec3(0, 1, 0))
        translation = glm.translate(glm.mat4(1), glm.vec3(*start))
        self.model_matrix = translation * rotation

        # Generating the mesh
        vertex_attributes = []
        z1, z2 = 0, glm.length(direction)
        for i in range(circular_segments):
            angle = np.pi * 2 * i / circular_segments
            next_angle = np.pi * 2 * (i+1) / circular_segments

            x1, y1 = np.cos(angle), np.sin(angle)
            x2, y2 = np.cos(next_angle), np.sin(next_angle)

            vertex_attributes.extend([
                # Wall line
                x1 * start_radius, y1 * start_radius, z1,
                x1 * end_radius,   y1 * end_radius,   z2,

                # Start floor line
                x1 * start_radius, y1 * start_radius, z1,
                x2 * start_radius, y2 * start_radius, z1,

                # End floor line
                x1 * end_radius, y1 * end_radius, z2,
                x2 * end_radius, y2 * end_radius, z2,
            ])

        self.vertex_attributes = np.array(vertex_attributes, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_attributes, GL_STATIC_DRAW)
        float_size = self.vertex_attributes.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * float_size, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBindVertexArray(0)

    def draw(self, delta_time: float):
        self.shader.use()
        self.shader.set_color((1, 0, 0, 1))

        self.shader.set_model_matrix(self.model_matrix)

        # Sending the draw call
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, len(self.vertex_attributes) // 3)
        glBindVertexArray(0)

