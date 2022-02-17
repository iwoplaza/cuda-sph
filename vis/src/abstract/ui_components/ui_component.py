from dataclasses import dataclass
from typing import Tuple
from ..component import Component


@dataclass
class BoundingBox:
    min: Tuple[int, int]
    max: Tuple[int, int]


class UIComponent(Component):
    def __init__(self, position: Tuple[int, int]) -> None:
        super().__init__()

        self._position = position

    def set_position(self, position: Tuple[int, int]):
        self._position = position

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError()
