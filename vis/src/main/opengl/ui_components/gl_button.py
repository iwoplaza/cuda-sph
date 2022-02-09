from typing import Tuple
from OpenGL.GL import *
import glm
from vis.src.main.abstract.ui_components import Button
from vis.src.main.abstract.ui_components import Font
from vis.src.main.opengl.common_shaders import CommonShaders
from ..mesh import make_rectangle_mesh


class GLButton(Button):
    def __init__(self, font: Font, pos: Tuple[float, float], text: str):
        super().__init__(font=font, pos=pos, text=text)

        self.shader = CommonShaders.UI_SOLID
        self.text_width = font.estimate_width(text)

        padding_x, padding_y = 15, 5

        self.panel = make_rectangle_mesh((0, 0), (self.text_width + padding_x*2, font.font_size + padding_y * 2))
        self.text_origin = (pos[0] + padding_x, pos[1] + font.font_size - int(font.font_size / 5) + padding_y)

    def draw(self, delta_time: float):
        # Background
        self.shader.use()
        model_view = glm.translate(glm.mat4(), glm.vec3(*self.pos, 0))
        self.shader.set_model_matrix(model_view)

        self.panel.draw()

        # Text
        self.font.use()
        self.font.draw_text(self.text, self.text_origin)
