from typing import Tuple
from vis.src.abstract.ui_components import Button
from vis.src.abstract import Font


class UIComponentFactory:
    def create_button(self, font: Font, pos: Tuple[int, int], label: str, click_command=None) -> Button:
        raise NotImplementedError()