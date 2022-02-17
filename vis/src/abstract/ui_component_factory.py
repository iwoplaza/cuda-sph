from typing import Tuple
from vis.src.abstract.ui_components import Button
from vis.src.abstract import Font
from vis.src.abstract.ui_components.stack_layout import StackLayout


class UIComponentFactory:
    def create_button(self, font: Font, pos: Tuple[int, int], label: str, click_command=None) -> Button:
        raise NotImplementedError()

    def create_stack_layout(self, position: Tuple[int, int], spacing: int) -> StackLayout:
        return StackLayout(position, spacing)
