from vis.src.main.abstract import Layer, UILayerContext
from vis.src.main.commands import PositionCamera


class MainUILayer(Layer):
    def __init__(self, fct: UILayerContext):
        super().__init__()

        self.font = fct.create_font("assets/FreeSans.ttf")

        self.test_button = fct.create_button(self.font, (25, 50), 'Test 1')
        self.test_button_2 = fct.create_button(self.font, (110, 50), 'Reset camera',
                                               lambda: fct.dispatch_command(
                                                   PositionCamera(position=(0, 0, 0), yaw=0, pitch=0)
                                               ))

        self.add(self.test_button)
        self.add(self.test_button_2)
