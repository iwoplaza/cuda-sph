from .ui_screen import UIScreen


class Window:
    def __init__(self, title, width=800, height=600):
        self.title = title
        self.width = width
        self.height = height
        self.current_screen = None

    def draw_current_screen(self):
        if self.current_screen is not None:
            self.current_screen.draw()

    def show_screen(self, screen: UIScreen):
        self.current_screen = screen

    def run(self):
        raise NotImplementedError()
