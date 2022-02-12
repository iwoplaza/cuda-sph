import numpy as np

from vis.src.abstract import Layer, SceneLayerContext
from vis.src.playback_management import PlaybackManager


class ViewportLayer(Layer):
    def __init__(self, fct: SceneLayerContext, playback_manager: PlaybackManager):
        super().__init__()

        self.playback_manager = playback_manager

        self.point_field = fct.create_point_field((0, 0, 0), (1, 1, 1))
        self.add(self.point_field)

        self.pipes = [
            (2, 1),
            (2, 1),
            (3, 2),
            (1, 1.5),
            (2, 1),
        ]

        self.pipe_segments = []
        point = 0
        for i in range(len(self.pipes)):
            pipe = self.pipes[i]
            next_pipe = self.pipes[(i + 1) % len(self.pipes)]

            seg = fct.create_wire_cylinder((0, 0, point), (0, 0, point + pipe[0]), pipe[1], next_pipe[1])
            self.pipe_segments.append(seg)
            self.add(seg)

            point += pipe[0]

        self.camera = fct.create_camera((-5, 15, -5), yaw=np.pi*3/4, pitch=np.pi*0.22)
        self.add(self.camera)

        self.points = None

    def update(self, delta_time: float):
        self.playback_manager.update(delta_time)

        self.points = self.playback_manager.get_current_state()

        self.point_field.set_point_positions(self.points)
