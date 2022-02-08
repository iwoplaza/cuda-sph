from vis.src.main.abstract import Layer, SceneComponentFactory


class ViewportLayer(Layer):
    component_factory: SceneComponentFactory

    def __init__(self, fct: SceneComponentFactory):
        super().__init__()

        self.component_factory = fct

        self.point_field = fct.create_point_field((0, 0, 0), (1, 1, 1))
        self.point_field.set_point_positions([
            (0, 0, -1),
            (1, 0, -1),
            (0, 1, -1),
            (0, 0, 1),
        ])

        self.test_cube = fct.create_cube((0, 0, -5), (1, 1, 1))

    def draw(self):
        self.point_field.draw()
        self.test_cube.draw()
