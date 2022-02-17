from .layer import Layer
from vis.src.abstract.component_database import ComponentDatabase


class Window:
    _component_database: ComponentDatabase

    def __init__(self, title, width=800, height=600):
        self.title = title
        self.width = width
        self.height = height
        self._layers = []
        self._component_database = ComponentDatabase()

    def add_layer(self, layer: Layer):
        layer.setup(self._component_database)
        self._layers.append(layer)

    def perform_command(self, command):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
