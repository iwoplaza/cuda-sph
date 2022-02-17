import os

import numpy as np

import config
from vis.src.abstract import Layer, LayerContext, UIComponentFactory
from vis.src.command.commands import PositionCamera
from vis.src.fps_counter import FPSCounter
from vis.src.playback_management import PlaybackManager


class MainUILayer(Layer):
    def __init__(self, fct: UIComponentFactory, ctx: LayerContext, playback_manager: PlaybackManager):
        super().__init__()

        self.__playback_manager = playback_manager
        self.__playback_manager.set_looping(True)
        self.__showing_stats = False
        self.__fps_counter = FPSCounter()

        self.font = ctx.create_font(os.path.join(config.ROOT_PROJ_DIRNAME, config.ASSETS_DIRNAME, "Roboto-Regular.ttf"))

        self.toolbar_layout = fct.create_stack_layout((10, 10), 10)
        self.add(self.toolbar_layout)

        # Buttons
        toggle_stats_btn = fct.create_button(self.font, (0, 0), 'Toggle stats', self.__toggle_show_stats)
        self.toolbar_layout.add(toggle_stats_btn)

        reset_camera_btn = fct.create_button(self.font, (0, 0), 'Reset camera',
                                                  lambda: ctx.invoke_command(
                                                   PositionCamera('main-camera', position=(-5, 15, -5), yaw=np.pi*3/4,
                                                                  pitch=np.pi*0.22)
                                               ))
        self.toolbar_layout.add(reset_camera_btn)

        self.play_btn = fct.create_button(self.font, (0, 0), 'Play', self.__toggle_playback)
        self.toolbar_layout.add(self.play_btn)

    def __toggle_show_stats(self):
        self.__showing_stats = not self.__showing_stats

    def __toggle_playback(self):
        new_state = 1 if self.__playback_manager.get_state() == 0 else 0
        self.__playback_manager.set_state(new_state)

        self.play_btn.set_label('Play' if new_state == 0 else 'Pause')

    def draw(self, delta_time: float):
        super().draw(delta_time)

        if self.__showing_stats:
            self.font.use()
            fps = self.__fps_counter.get_fps(delta_time)
            self.font.draw_text(f"FPS: {fps}", (10, 60))
            self.font.draw_text("Time: %.2f" % self.__playback_manager.get_time_elapsed(), (1, 60 + 40))
