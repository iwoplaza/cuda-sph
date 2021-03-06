from typing import List
from .component import Component
from .component_database import ComponentDatabase


class Layer:
    def __init__(self):
        self.components: List[Component] = []
        self._component_database: ComponentDatabase = None

    def _update(self, delta_time: float):
        pass

    def setup(self, component_database):
        self._component_database = component_database

    def register(self, identifier: str, component: Component):
        self._component_database.register(identifier, component)

    def add(self, component: Component):
        self.components.append(component)

    def draw(self, delta_time: float):
        self._update(delta_time)

        for c in self.components:
            c.draw(delta_time)

    def on_mouse_move(self, x: int, y: int) -> None:
        for c in self.components:
            c.on_mouse_move(x, y)

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        captured = False

        for c in self.components:
            captured |= c.on_mouse_btn_pressed(x, y, button)

        return captured

    def on_mouse_btn_released(self, x: int, y: int, button: int) -> None:
        for c in self.components:
            c.on_mouse_btn_released(x, y, button)

    def on_key_pressed(self, key: bytes) -> bool:
        captured = False

        for c in self.components:
            captured |= c.on_key_pressed(key)

        return captured

    def on_key_released(self, key: bytes) -> None:
        for c in self.components:
            c.on_key_released(key)
