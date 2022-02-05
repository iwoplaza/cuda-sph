from src.main.abstract import UIScreen
from src.main.abstract.components import ComponentFactory


class VisualScreen(UIScreen):
    component_factory: ComponentFactory

    def __init__(self, component_factory: ComponentFactory):
        super().__init__()

        self.component_factory = component_factory
        self.font = self.component_factory.create_font("assets/FreeSans.ttf")

    def draw(self):
        self.font.use()
        self.font.draw_text("Hello", (25, 50))

