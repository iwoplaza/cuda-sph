from .font import Font


class ComponentFactory:
    def create_font(self, path: str, font_size: int = 48) -> Font:
        raise NotImplementedError()

