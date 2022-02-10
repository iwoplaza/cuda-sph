import numpy as np

from vis.src.main.abstract import Layer, UILayerContext
from vis.src.main.commands import PositionCamera
from vis.src.main.fps_counter import FPSCounter


class MainUILayer(Layer):
    def __init__(self, fct: UILayerContext):
        super().__init__()

        self.__showing_stats = False
        self.__fps_counter = FPSCounter()

        self.font = fct.create_font("assets/Roboto-Regular.ttf")

        self.test_button = fct.create_button(self.font, (10, 10), 'Toggle stats', self.__toggle_show_stats)
        self.test_button_2 = fct.create_button(self.font, (145, 10), 'Reset camera',
                                               lambda: fct.dispatch_command(
                                                   PositionCamera(position=(-5, 15, -5), yaw=np.pi*3/4, pitch=np.pi*0.22)
                                               ))

        self.add(self.test_button)
        self.add(self.test_button_2)

    def __toggle_show_stats(self):
        self.__showing_stats = not self.__showing_stats

    def draw(self, delta_time: float):
        super().draw(delta_time)

        if self.__showing_stats:
            self.font.use()
            fps = self.__fps_counter.get_fps(delta_time)
            self.font.draw_text(f"FPS: {fps}", (10, 60))
