from typing import Tuple
import numpy
from freetype import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import math
import glm
from vis.src.abstract.ui_components import Font
from vis.src.color import Color
from vis.src.opengl.common_shaders import CommonShaders


class GLFont(Font):
    def __init__(self, font_path: str, font_size: int = 48):
        # Setting up the shared font shader
        super().__init__(font_size)

        self.shader = CommonShaders.FONT
        self.base, self.texid = 0, 0
        self.characters = []

        vertex_attributes = numpy.array([
            # x   y  u  v
            0, -1, 0, 0,
            0, 0, 0, 1,
            1, 0, 1, 1,
            0, -1, 0, 0,
            1, 0, 1, 1,
            1, -1, 1, 0
        ], dtype=numpy.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes, GL_STATIC_DRAW)
        float_size = vertex_attributes.itemsize
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 4 * float_size, None)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 4 * float_size, c_void_p(2 * float_size))
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindVertexArray(0)

        self.__make_font(font_path, self.font_size)

    def __make_font(self, filename, font_size):
        face = Face(filename)
        face.set_pixel_sizes(0, font_size)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glActiveTexture(GL_TEXTURE0)

        for c in range(128):
            face.load_char(chr(c))
            glyph = face.glyph
            bitmap = glyph.bitmap
            size = bitmap.width, bitmap.rows
            bearing = glyph.bitmap_left, glyph.bitmap_top
            advance = glyph.advance.x

            # create glyph texture
            texObj = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texObj)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, *size, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.buffer)

            self.characters.append((texObj, size, bearing, advance))

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)

    def use(self):
        self.shader.use()

    def draw_text(self,
                  text: str,
                  pos: Tuple[float, float],
                  color: Color = (1, 1, 1, 1),
                  scale: float = 1,
                  direction: Tuple[float, float] = (1, 0)
                  ):
        self.shader.use()
        glUniform3f(2, color[0], color[1], color[2])

        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.vao)
        angle_rad = math.atan2(direction[1], direction[0])
        rotateM = glm.rotate(glm.mat4(1), angle_rad, glm.vec3(0, 0, 1))
        transOriginM = glm.translate(glm.mat4(1), glm.vec3(*pos, 0))

        char_x = 0
        for c in text:
            c = ord(c)
            ch = self.characters[c]
            w, h = ch[1][0] * scale, ch[1][1] * scale
            xrel, yrel = char_x + ch[2][0] * scale, (ch[1][1] - ch[2][1]) * scale
            char_x += (ch[3] >> 6) * scale
            scaleM = glm.scale(glm.mat4(1), glm.vec3(w, h, 1))
            transRelM = glm.translate(glm.mat4(1), glm.vec3(xrel, yrel, 0))
            modelM = transOriginM * rotateM * transRelM * scaleM

            self.shader.set_model_matrix(modelM)
            glBindTexture(GL_TEXTURE_2D, ch[0])
            glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def estimate_width(self, text: str):
        char_x = 0
        for c in text:
            c = ord(c)
            ch = self.characters[c]
            char_x += (ch[3] >> 6)

        return char_x
