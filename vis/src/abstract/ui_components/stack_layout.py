from typing import Tuple, List
from vis.src.abstract.ui_components.ui_component import UIComponent
from .ui_component import UIComponent, BoundingBox


class StackLayout(UIComponent):
    def __init__(self, position: Tuple[int, int], spacing: float = 0) -> None:
        super().__init__(position)

        self.__children: List[UIComponent] = []
        self.__spacing = spacing
        self.__bounding_box = BoundingBox((0, 0), (0, 0))
        self.update_layout()

    def update_layout(self):
        max_child_height = 0
        offset_x = self._position[0]

        for child in self.__children:
            child_bounds = child.get_bounding_box()
            child_width = child_bounds.max[0] - child_bounds.min[0]
            child_height = child_bounds.max[1] - child_bounds.min[1]

            child.set_position((offset_x, self._position[1]))

            max_child_height = max(max_child_height, child_height)
            offset_x += child_width + self.__spacing

        min_x = self._position[0]
        max_x = self._position[0] + max(0, offset_x - self.__spacing)

        min_y = self._position[1]
        max_y = self._position[1] + max_child_height

        self.__bounding_box = BoundingBox(
            min=(min_x, min_y),
            max=(max_x, max_y)
        )

    def add(self, child: UIComponent):
        self.__children.append(child)
        self.update_layout()

    def on_mouse_move(self, x: int, y: int) -> None:
        for child in self.__children:
            child.on_mouse_move(x, y)

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        captured = False
        for child in self.__children:
            captured |= child.on_mouse_btn_pressed(x, y, button)

        return captured

    def on_mouse_btn_released(self, x: int, y: int, button: int) -> None:
        for child in self.__children:
            child.on_mouse_btn_released(x, y, button)

    def on_key_pressed(self, key: bytes) -> bool:
        captured = False
        for child in self.__children:
            captured |= child.on_key_pressed(key)

        return captured

    def on_key_released(self, key: bytes) -> None:
        for child in self.__children:
            child.on_key_released(key)

    def draw(self, delta_time: float):
        super().draw(delta_time)

        for child in self.__children:
            child.draw(delta_time)

    def get_bounding_box(self):
        return self.__bounding_box
