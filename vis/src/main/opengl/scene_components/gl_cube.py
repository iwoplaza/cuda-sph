from vis.src.main.abstract.scene_components import Cube
from vis.src.main.vector import Vec3f
from vis.src.main.opengl.common_shaders import CommonShaders
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import glm


class GLCube(Cube):
    def __init__(self, origin: Vec3f, scale: Vec3f):
        super().__init__(origin=origin, scale=scale)

        self.shader = CommonShaders.SOLID

        vertex_attributes = np.array([
            # x y z

            # Back
            -1, 1, -1,
            1, 1, -1,
            -1, -1, -1,
            -1, -1, -1,
            1, 1, -1,
            1, -1, -1,

            # Front
            -1, 1, 1,
            1, 1, 1,
            -1, -1, 1,
            -1, -1, 1,
            1, 1, 1,
            1, -1, 1,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes, GL_STATIC_DRAW)
        float_size = vertex_attributes.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * float_size, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBindVertexArray(0)

        self.angle = 0

    def draw(self, deltaTime: float):
        self.shader.use()

        self.angle += np.pi / 4 * deltaTime  # quarter-turn per second

        translation = glm.translate(glm.mat4(1), glm.vec3(*self.origin))
        rotation = glm.rotate(glm.mat4(1), self.angle, glm.vec3(0, 1, 0))
        model_matrix = translation * rotation

        self.shader.set_model_matrix(model_matrix)

        # Sending the draw call
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6 * 2)
        glBindVertexArray(0)
