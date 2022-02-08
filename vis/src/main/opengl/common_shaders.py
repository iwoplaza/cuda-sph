from .shader import Shader, FileShader


class CommonShaders:
    # UI
    FONT: Shader = None
    UI_SOLID: Shader = None

    # Viewport
    SOLID: Shader = None
    POINT_FIELD: Shader = None

    @classmethod
    def register_ui_shaders(cls) -> list[Shader]:
        shaders = []

        cls.FONT = FileShader('text_shader', 'text_shader')
        shaders.append(cls.FONT)

        cls.UI_SOLID = FileShader('ui_solid_shader', 'ui_solid_shader')
        shaders.append(cls.UI_SOLID)

        return shaders

    @classmethod
    def register_scene_shaders(cls) -> list[Shader]:
        shaders = []

        cls.SOLID = FileShader('solid', 'solid')
        shaders.append(cls.SOLID)

        cls.POINT_FIELD = FileShader('point_field', 'point_field')
        shaders.append(cls.POINT_FIELD)

        return shaders
