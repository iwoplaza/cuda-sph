from typing import Tuple
from vis.src.abstract.font import Font
from .ui_component import UIComponent


class Button(UIComponent):
    def __init__(self, font: Font, pos: Tuple[int, int], text: str, click_command=None):
        super().__init__(pos)

        self._font = font
        self._label = text
        self.click_command = click_command

        self._text_width = font.estimate_width(text)

    def set_label(self, text: str):
        self._label = text
        self._text_width = self._font.estimate_width(text)

    def draw(self, delta_time: float):
        raise NotImplementedError()
