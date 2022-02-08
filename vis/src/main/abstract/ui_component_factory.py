from typing import Tuple
from vis.src.main.abstract.ui_components import Font, Button


class UIComponentFactory:
    def create_font(self, path: str, font_size: int = 20) -> Font:
        raise NotImplementedError()

    def create_button(self, font: Font, pos: Tuple[int, int], label: str) -> Button:
        raise NotImplementedError()
