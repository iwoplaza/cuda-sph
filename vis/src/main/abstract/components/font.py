from typing import Tuple
from src.main.color import Color


class Font:
    def __init__(self, font_size=48):
        self.font_size = font_size

    def use(self):
        pass

    def draw_text(self,
                  text: str,
                  pos: Tuple[float, float],
                  color: Color = (1, 1, 1, 1),
                  scale: float = 1,
                  direction: Tuple[float, float] = (1, 0)
                  ):
        pass
