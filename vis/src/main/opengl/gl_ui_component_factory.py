from typing import Tuple
from vis.src.main.abstract import UIComponentFactory
from vis.src.main.abstract.ui_components import Font
from .ui_components import GLFont, GLButton
from .gl_window import GLWindow


class GLUIComponentFactory(UIComponentFactory):
    window: GLWindow

    def __init__(self, window: GLWindow):
        self.window = window

    def create_font(self, path: str, font_size: int = 16):
        return GLFont(path, font_size)

    def create_button(self, font: Font, pos: Tuple[int, int], label: str):
        return GLButton(font=font, pos=pos, text=label)
