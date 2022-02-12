from typing import List
from vis.src.main.vector import Vec3f
from vis.src.main.abstract import Layer, SceneLayerContext
import numpy as np


def generate_random_points(amount: int) -> List[Vec3f]:
    points = []

    x_range = (0.0, 10.0)
    y_range = (0.0, 10.0)
    z_range = (0.0, 10.0)

    for i in range(amount):
        points.append((
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range),
        ))

    return points


def move_points(points: List[Vec3f]):
    for i in range(len(points)):
        points[i] = (
            points[i][0] + np.random.uniform(-1, 1) * 0.01,
            points[i][1] + np.random.uniform(-1, 1) * 0.01,
            points[i][2] + np.random.uniform(-1, 1) * 0.01
        )


class ViewportLayer(Layer):
    def __init__(self, fct: SceneLayerContext):
        super().__init__()

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

        self.points = generate_random_points(100000)

    def draw(self, delta_time: float):
        # move_points(self.points)
        self.point_field.set_point_positions(self.points)

        super().draw(delta_time)
