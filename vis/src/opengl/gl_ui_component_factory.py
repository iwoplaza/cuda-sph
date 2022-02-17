from typing import Tuple
from vis.src.abstract import UIComponentFactory
from vis.src.abstract import Font
from .ui_components import GLButton


class GLUIComponentFactory(UIComponentFactory):
    def create_button(self, font: Font, pos: Tuple[int, int], label: str, click_command=None):
        return GLButton(font=font, pos=pos, text=label, click_command=click_command)
