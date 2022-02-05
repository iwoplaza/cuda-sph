# TODO
# - Facade (Window creation, OpenGL)
# - Flyweight (particles, coloring)
# - Chain of responsibility (UI events)
# - Command (UI)
# - Mediator (UI)


from src.main.opengl import GLWindow
from src.main.opengl import GLComponentFactory
from src.main.screen import VisualScreen


if __name__ == '__main__':
    window = GLWindow(title="Visualization - Smoothed Particle Hydrodynamics")
    component_factory = GLComponentFactory(window)

    window.show_screen(VisualScreen(component_factory))
    window.run()
