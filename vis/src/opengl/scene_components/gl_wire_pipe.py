import math
import glm
import numpy as np
from OpenGL.GL import *

from common.data_classes import Pipe
from vis.src.abstract.scene_components.wire_pipe import WirePipe
from vis.src.vector import Vec3f
from vis.src.opengl.common_shaders import CommonShaders


class GLWirePipe(WirePipe):
    def __init__(self, pipe: Pipe):
        """
        Assuming that pipe is already defined in world space,
        so no translation/rotation/scale needed
        """
        super().__init__()
        self.__pipe = pipe
        self.shader = CommonShaders.SOLID
        self.__vertex_data = self.__generate_vertex_data()
        self.__vao = self.__init_vao()

    def __generate_vertex_data(self):
        data = []

        circular_lines_count = 10
        angle_step = np.pi * 2 / float(circular_lines_count)

        for segment in self.__pipe.segments:
            xs = [segment.start_point[0], segment.start_point[0] + segment.length]
            rs = [segment.start_radius, segment.end_radius]
            for i in range(2):
                for j in range(circular_lines_count):
                    angle = angle_step * j
                    next_angle = angle_step * (j + 1)
                    y1, z1 = np.cos(angle), np.sin(angle)
                    y2, z2 = np.cos(next_angle), np.sin(next_angle)
                    data.extend([
                        xs[i], y1 * rs[i], z1,
                        xs[i], y2 * rs[i], z2
                    ])
                    if i == 0:
                        data.extend([
                            xs[i], y1 * rs[i], z1,
                            xs[i+1], y1 * rs[i+1], z1
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

    def draw(self, delta_time: float):
        self.shader.use()
        self.shader.set_color((1, 0, 0, 1))

        self.shader.set_model_matrix(glm.mat4(1.0))

        # Sending the draw call
        glBindVertexArray(self.__vao)
        glDrawArrays(GL_LINES, 0, len(self.__vertex_data) // 3)
        glBindVertexArray(0)