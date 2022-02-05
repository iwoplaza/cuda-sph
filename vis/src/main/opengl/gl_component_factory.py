from typing import Tuple
from src.main.abstract.components import Font, ComponentFactory
from .components import GLFont, GLButton
from .gl_window import GLWindow


class GLComponentFactory(ComponentFactory):
    window: GLWindow

    def __init__(self, window: GLWindow):
        self.window = window

    def create_font(self, path: str, font_size: int = 16):
        return GLFont(path, font_size)

    def create_button(self, font: Font, pos: Tuple[int, int], label: str):
        return GLButton(font=font, pos=pos, text=label)
