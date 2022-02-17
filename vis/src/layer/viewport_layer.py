import numpy as np

import config
from common.data_classes import SimulationParameters
from vis.src.abstract import Layer, SceneLayerContext
from vis.src.playback_management import PlaybackManager


class ViewportLayer(Layer):
    def __init__(self, fct: SceneLayerContext, playback_manager: PlaybackManager, params: SimulationParameters):
        super().__init__()
        self.params = params
        print(params.space_size)
        self.playback_manager = playback_manager

        self.camera = fct.create_camera(self.params.space_size * 1.2, yaw=np.pi / -4.0, pitch=np.pi / 4.0)
        self.add(self.camera)

        self.particles = fct.create_point_field((0, 0, 0), (1, 1, 1))
        self.add(self.particles)

        # self.cube = fct.create_wire_cube((0, 0, 0), params.space_size)
        # self.add(self.cube)
        if config.SIM_MODE == 'PIPE':
            self.pipe = fct.create_wire_pipe(params.pipe)
            self.add(self.pipe)

    def _update(self, delta_time: float):
        self.playback_manager.update(delta_time)
        # print(self.camera.)
        self.particles.set_point_positions(self.playback_manager.get_current_state())



