import numpy as np

from vis.src.abstract import Layer, UILayerContext
from vis.src.commands import PositionCamera
from vis.src.fps_counter import FPSCounter
from vis.src.playback_management import PlaybackManager


class MainUILayer(Layer):
    def __init__(self, ctx: UILayerContext, playback_manager: PlaybackManager):
        super().__init__()

        self.__playback_manager = playback_manager
        self.__playback_manager.set_looping(True)
        self.__showing_stats = False
        self.__fps_counter = FPSCounter()

        self.font = ctx.create_font("vis/assets/Roboto-Regular.ttf")

        self.test_button = ctx.create_button(self.font, (10, 10), 'Toggle stats', self.__toggle_show_stats)
        self.test_button_2 = ctx.create_button(self.font, (145, 10), 'Reset camera',
                                               lambda: ctx.dispatch_command(
                                                   PositionCamera(position=(-5, 15, -5), yaw=np.pi*3/4, pitch=np.pi*0.22)
                                               ))

        self.add(self.test_button)
        self.add(self.test_button_2)

    def __toggle_show_stats(self):
        self.__showing_stats = not self.__showing_stats
        self.__playback_manager.set_state(1)

    def draw(self, delta_time: float):
        super().draw(delta_time)

        if self.__showing_stats:
            self.font.use()
            fps = self.__fps_counter.get_fps(delta_time)
            self.font.draw_text(f"FPS: {fps}", (10, 60))
            self.font.draw_text("Time: %.2f" % self.__playback_manager.get_time_elapsed(), (1, 60 + 40))
