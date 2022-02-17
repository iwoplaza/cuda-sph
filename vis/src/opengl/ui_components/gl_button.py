from typing import Tuple
import glm
from vis.src.abstract.ui_components import BoundingBox, Button
from vis.src.abstract import Font
from vis.src.opengl.common_shaders import CommonShaders
from ..mesh import make_rectangle_mesh


class GLButton(Button):
    def __init__(self, font: Font, pos: Tuple[int, int], text: str, click_command=None):
        super().__init__(font=font, pos=pos, text=text, click_command=click_command)

        self.padding_x = 15
        self.padding_y = 5

        self.__update_bounding_box()

        self.shader = CommonShaders.UI_SOLID
        self.panel = make_rectangle_mesh((0, 0), (1, 1))

        # Visual state
        self.active = False

    def __update_bounding_box(self):
        self.size = (self._text_width + self.padding_x * 2, self._font.font_size + self.padding_y * 2)

        self.text_origin = (
            self._position[0] + self.padding_x,
            self._position[1] + self._font.font_size - int(self._font.font_size / 5) + self.padding_y
        )

        self.__bounding_box = BoundingBox(
            min=(self._position[0], self._position[1]),
            max=(self._position[0] + self.size[0], self._position[1] + self.size[1])
        )

    def set_label(self, text: str):
        super().set_label(text)
        self.__update_bounding_box()

    def set_position(self, position: Tuple[int, int]):
        super().set_position(position)
        self.__update_bounding_box()

    def get_bounding_box(self) -> BoundingBox:
        return self.__bounding_box

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        if (x < self._position[0] or x > self._position[0] + self.size[0] or
                y < self._position[1] or y > self._position[1] + self.size[1]):
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
        model_view = glm.translate(glm.mat4(), glm.vec3(*self._position, 0))
        model_view = glm.scale(model_view, glm.vec3(*self.size, 1))
        self.shader.set_model_matrix(model_view)

        if not self.active:
            self.shader.set_color((0.1, 0.1, 0.1, 1))
        else:
            self.shader.set_color((0.2, 0.2, 0.2, 1))

        self.panel.draw()

        # Text
        self._font.use()
        self._font.draw_text(self._label, self.text_origin)
