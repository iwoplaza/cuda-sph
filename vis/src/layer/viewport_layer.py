import numpy as np

from common.data_classes import SimulationParameters, SimulationState
from vis.src.abstract import Layer, LayerContext, SceneComponentFactory
from vis.src.playback_management import PlaybackManager


class ViewportLayer(Layer):
    def __init__(self,
                 fct: SceneComponentFactory,
                 ctx: LayerContext,
                 playback_manager: PlaybackManager,
                 params: SimulationParameters):
        super().__init__()
        self.params = params
        self.playback_manager = playback_manager

        self.camera = fct.create_camera(self.params.space_size * 1.2, yaw=np.pi / -4.0, pitch=np.pi / 4.0)
        self.add(self.camera)

        self.particles = fct.create_point_field((0, 0, 0), (1, 1, 1))
        self.add(self.particles)

        if len(params.pipe.segments) > 0:
            self.pipe = fct.create_wire_pipe(params.pipe)
            self.add(self.pipe)
        else:
            self.cube = fct.create_wire_cube((0, 0, 0), params.space_size)
            self.add(self.cube)

    def setup(self, component_database):
        super().setup(component_database)

        self.register('main-camera', self.camera)

    def _update(self, delta_time: float):
        self.playback_manager.update(delta_time)
        state: SimulationState = self.playback_manager.get_current_data()
        self.particles.set_point_positions(state.position)
        self.particles.set_point_densities(state.density)
