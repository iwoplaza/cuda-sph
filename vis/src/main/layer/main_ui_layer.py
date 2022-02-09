from vis.src.main.abstract import Layer, UIComponentFactory


class MainUILayer(Layer):
    def __init__(self, fct: UIComponentFactory):
        super().__init__()

        self.font = fct.create_font("assets/FreeSans.ttf")

        self.test_button = fct.create_button(self.font, (25, 50), 'Test 1')
        self.test_button_2 = fct.create_button(self.font, (110, 50), 'Hello world')

        self.add(self.test_button)
        self.add(self.test_button_2)
