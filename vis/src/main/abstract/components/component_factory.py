from typing import Tuple
from .font import Font


class ComponentFactory:
    def create_font(self, path: str, font_size: int = 20) -> Font:
        raise NotImplementedError()

    def create_button(self, font: Font, pos: Tuple[int, int], label: str):
        raise NotImplementedError()
