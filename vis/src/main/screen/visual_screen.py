from src.main.abstract import UIScreen
from src.main.abstract.components import ComponentFactory


class VisualScreen(UIScreen):
    component_factory: ComponentFactory

    def __init__(self, component_factory: ComponentFactory):
        super().__init__()

        self.component_factory = component_factory
        self.font = self.component_factory.create_font("assets/FreeSans.ttf")

        self.test_button = self.component_factory.create_button(self.font, (25, 50), 'Test 1')
        self.test_button_2 = self.component_factory.create_button(self.font, (110, 50), 'Hello world')

    def draw(self):
        self.test_button.draw()
        self.test_button_2.draw()
