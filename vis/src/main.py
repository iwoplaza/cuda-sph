# TODO
# - Facade (Window creation, OpenGL)
# - Flyweight (particles, coloring)
# - Chain of responsibility (UI events)
# - Command (UI)
# - Mediator (UI)
import config
from common.serializer.loader import Loader
from vis.src.opengl import GLWindow
from vis.src.opengl import GLUILayerContext, GLSceneLayerContext
from vis.src.layer import MainUILayer, ViewportLayer
from vis.src.playback_management import PlaybackManager
from vis.src.playback_management import LazySPHLoadingStrategy


if __name__ == '__main__':
    window = GLWindow(title="Visualization - Smoothed Particle Hydrodynamics")
    scene_layer_context = GLSceneLayerContext(window)
    ui_layer_context = GLUILayerContext(window)

    params = Loader('out').load_simulation_parameters()
    loading_strategy = LazySPHLoadingStrategy(config.OUT_DIRNAME)
    playback_manager = PlaybackManager(loading_strategy)

    window.add_layer(ViewportLayer(scene_layer_context, playback_manager, params))
    window.add_layer(MainUILayer(ui_layer_context, playback_manager))
    window.run()
