from vis.src.abstract import LayerContext
from .gl_font import GLFont
from .gl_window import GLWindow


class GLLayerContext(LayerContext):
    __window: GLWindow

    def __init__(self, window: GLWindow):
        self.__window = window

    def invoke_command(self, command) -> None:
        self.__window.perform_command(command)

    def create_font(self, path: str, font_size: int = 18):
        return GLFont(path, font_size)
