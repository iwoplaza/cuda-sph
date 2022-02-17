from typing import Tuple
from vis.src.abstract import UILayerContext
from vis.src.abstract.ui_components import Font
from .ui_components import GLFont, GLButton
from .gl_window import GLWindow


class GLUILayerContext(UILayerContext):
    __window: GLWindow

    def __init__(self, window: GLWindow):
        self.__window = window

    def invoke_command(self, command) -> None:
        self.__window.perform_command(command)

    def create_font(self, path: str, font_size: int = 18):
        return GLFont(path, font_size)

    def create_button(self, font: Font, pos: Tuple[int, int], label: str, click_command=None):
        return GLButton(font=font, pos=pos, text=label, click_command=click_command)
