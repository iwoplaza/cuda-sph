from .layer import Layer


class Window:
    def __init__(self, title, width=800, height=600):
        self.title = title
        self.width = width
        self.height = height
        self._layers = []

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    def run(self):
        raise NotImplementedError()
