from typing import Tuple
from .font import Font


class Button:
    def __init__(self, font: Font, pos: Tuple[float, float], text: str):
        self.font = font
        self.pos = pos
        self.text = text

    def draw(self):
        raise NotImplementedError()
