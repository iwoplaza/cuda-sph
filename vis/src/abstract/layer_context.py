from vis.src.abstract.ui_components import Font


class LayerContext:
    def invoke_command(self, command) -> None:
        raise NotImplementedError()

    def create_font(self, path: str, font_size: int = 20) -> Font:
        raise NotImplementedError()
