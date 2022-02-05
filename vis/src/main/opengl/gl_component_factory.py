from src.main.abstract.components import ComponentFactory
from .components import GLFont
from .gl_window import GLWindow


class GLComponentFactory(ComponentFactory):
    window: GLWindow

    def __init__(self, window: GLWindow):
        self.window = window

    def create_font(self, path: str, font_size: int = 48):
        font = GLFont(path, font_size)
        self.window.register_shader(font.shader)

        return font
