from typing import Tuple
from .font import Font
from ..component import Component


class Button(Component):
    def __init__(self, font: Font, pos: Tuple[float, float], text: str):
        self.font = font
        self.pos = pos
        self.text = text

    def draw(self, delta_time: float):
        raise NotImplementedError()
