from .layer import Layer


class Window:
    def __init__(self, title, width=800, height=600):
        self.title = title
        self.width = width
        self.height = height
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def run(self):
        raise NotImplementedError()
