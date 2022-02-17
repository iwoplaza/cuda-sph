from typing import Tuple
from vis.src.abstract.font import Font
from ..component import Component


class Button(Component):
    def __init__(self, font: Font, pos: Tuple[float, float], text: str, click_command=None):
        self.font = font
        self.pos = pos
        self.text = text
        self.click_command = click_command

    def draw(self, delta_time: float):
        raise NotImplementedError()
