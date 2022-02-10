from typing import Tuple
from vis.src.main.abstract.ui_components import Font, Button


class UILayerContext:
    def dispatch_command(self, command) -> None:
        raise NotImplementedError()

    def create_font(self, path: str, font_size: int = 20) -> Font:
        raise NotImplementedError()

    def create_button(self, font: Font, pos: Tuple[int, int], label: str, click_command=None) -> Button:
        raise NotImplementedError()
