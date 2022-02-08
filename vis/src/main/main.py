# TODO
# - Facade (Window creation, OpenGL)
# - Flyweight (particles, coloring)
# - Chain of responsibility (UI events)
# - Command (UI)
# - Mediator (UI)


from vis.src.main.opengl import GLWindow
from vis.src.main.opengl import GLUIComponentFactory, GLSceneComponentFactory
from vis.src.main.layer import MainUILayer, ViewportLayer


if __name__ == '__main__':
    window = GLWindow(title="Visualization - Smoothed Particle Hydrodynamics")
    scene_component_factory = GLSceneComponentFactory(window)
    ui_component_factory = GLUIComponentFactory(window)

    window.add_layer(ViewportLayer(scene_component_factory))
    window.add_layer(MainUILayer(ui_component_factory))
    window.run()
