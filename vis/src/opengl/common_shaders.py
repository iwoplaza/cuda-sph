from typing import List

from .shader import Shader, SceneShader, SolidShader, UISolidShader, load_shader_src_from_asset


class CommonShaders:
    # UI
    FONT: Shader = None
    UI_SOLID: UISolidShader = None

    # Viewport
    SOLID: SolidShader = None
    POINT_FIELD: SceneShader = None

    @classmethod
    def register_ui_shaders(cls) -> List[Shader]:
        shaders = []

        cls.FONT = Shader(**load_shader_src_from_asset('text_shader'))
        shaders.append(cls.FONT)

        cls.UI_SOLID = UISolidShader(**load_shader_src_from_asset('ui_solid_shader'))
        shaders.append(cls.UI_SOLID)

        return shaders

    @classmethod
    def register_scene_shaders(cls) -> List[SceneShader]:
        shaders = []

        cls.SOLID = SolidShader(**load_shader_src_from_asset('solid'))
        shaders.append(cls.SOLID)

        cls.POINT_FIELD = SceneShader(**load_shader_src_from_asset('point_field'))
        shaders.append(cls.POINT_FIELD)

        return shaders
