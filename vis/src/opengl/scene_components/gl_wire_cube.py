import glm
import numpy as np
from OpenGL.GL import *
from vis.src.abstract.scene_components.wire_cube import WireCube
from vis.src.opengl.common_shaders import CommonShaders
from vis.src.vector import Vec3f


class GLWireCube(WireCube):

    def __init__(self, position: Vec3f, scale: Vec3f):
        super(GLWireCube, self).__init__()
        scale_matrix = glm.scale(glm.mat4(1), glm.vec3(*scale))
        translation_matrix = glm.translate(glm.mat4(1), glm.vec3(*position))
        self.__model_matrix = translation_matrix * scale_matrix
        self.__shader = CommonShaders.SOLID
        self.__vertex_data = self.__generate_vertex_data()
        self.__vao = self.__init_vao()

    def draw(self, delta_time: float):
        self.__shader.use()
        self.__shader.set_color((1, 1, 0, 1))
        self.__shader.set_model_matrix(self.__model_matrix)

        glBindVertexArray(self.__vao)
        glDrawArrays(GL_LINES, 0, len(self.__vertex_data) // 3)
        glBindVertexArray(0)

    @staticmethod
    def __generate_vertex_data() -> np.ndarray:
        """
        Generate cube vertices, assuming they will be drawn directly from vbo (without ebo)
        and they will be drawn as line segments.
        """
        data = []

        for z in [0, 1]:  # front and back faces
            data.extend([
                # upper line
                0, 1, z,
                1, 1, z,

                # right line
                1, 1, z,
                1, 0, z,

                # down line
                1, 0, z,
                0, 0, z,

                # left line
                0, 0, z,
                0, 1, z
            ])

        # connect front and back
        for x in [0, 1]:
            for y in [0, 1]:
                data.extend([
                    x, y, 0,
                    x, y, 1
                ])

        return np.array(data, np.float32)

    def __init_vao(self):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.__vertex_data, GL_STATIC_DRAW)
        float_size = self.__vertex_data.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * float_size, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBindVertexArray(0)
        return vao

