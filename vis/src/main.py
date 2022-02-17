# TODO
# - Facade (Window creation, OpenGL)
# - Flyweight (particles, coloring)
# - Chain of responsibility (UI events)
# - Command (UI)
# - Mediator (UI)
import config
from common.serializer.loader import Loader
from vis.src.opengl import GLWindow
from vis.src.opengl import GLUIComponentFactory, GLSceneComponentFactory, GLLayerContext
from vis.src.layer import MainUILayer, ViewportLayer
from vis.src.playback_management import PlaybackManager
from vis.src.playback_management import LazySPHLoadingStrategy


if __name__ == '__main__':
    window = GLWindow(title="Visualization - Smoothed Particle Hydrodynamics")
    scene_component_factory = GLSceneComponentFactory(window)
    ui_component_factory = GLUIComponentFactory()
    layer_context = GLLayerContext(window)

    loader = Loader(config.OUT_DIRNAME)
    params = loader.load_simulation_parameters()
    loading_strategy = LazySPHLoadingStrategy(loader)
    playback_manager = PlaybackManager(loading_strategy)

    window.add_layer(ViewportLayer(scene_component_factory, layer_context, playback_manager, params))
    window.add_layer(MainUILayer(ui_component_factory, layer_context, playback_manager))
    window.run()
