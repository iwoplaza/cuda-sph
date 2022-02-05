from .shader import Shader, FileShader


class CommonShaders:
    FONT: Shader = None
    UI_SOLID: Shader = None

    @classmethod
    def register_common_shaders(cls) -> list[Shader]:
        shaders = []

        cls.FONT = FileShader('text_shader', 'text_shader')
        shaders.append(cls.FONT)

        cls.UI_SOLID = FileShader('ui_solid_shader', 'ui_solid_shader')
        shaders.append(cls.UI_SOLID)

        return shaders
