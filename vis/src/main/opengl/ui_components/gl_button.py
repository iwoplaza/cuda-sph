from typing import Tuple
import glm
from vis.src.main.abstract.ui_components import Button
from vis.src.main.abstract.ui_components import Font
from vis.src.main.opengl.common_shaders import CommonShaders
from ..mesh import make_rectangle_mesh


class GLButton(Button):
    def __init__(self, font: Font, pos: Tuple[float, float], text: str, click_command=None):
        super().__init__(font=font, pos=pos, text=text, click_command=click_command)

        padding_x, padding_y = 15, 5

        self.pos = pos
        self.text_width = font.estimate_width(text)
        self.text_origin = (pos[0] + padding_x, pos[1] + font.font_size - int(font.font_size / 5) + padding_y)
        self.size = (self.text_width + padding_x * 2, font.font_size + padding_y * 2)

        self.shader = CommonShaders.UI_SOLID
        self.click_command = click_command

        self.panel = make_rectangle_mesh((0, 0), self.size)

        # Visual state
        self.active = False

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        if (x < self.pos[0] or x > self.pos[0] + self.size[0] or
                y < self.pos[1] or y > self.pos[1] + self.size[1]):
            return False

        if self.click_command is not None:
            self.click_command()

        print('Clicked the button')

        self.active = True
        return True

    def on_mouse_btn_released(self, x: int, y: int, button: int) -> None:
        self.active = False

    def draw(self, delta_time: float):
        # Background
        self.shader.use()
        model_view = glm.translate(glm.mat4(), glm.vec3(*self.pos, 0))
        self.shader.set_model_matrix(model_view)

        if not self.active:
            self.panel.draw()

        # Text
        self.font.use()
        self.font.draw_text(self.text, self.text_origin)
