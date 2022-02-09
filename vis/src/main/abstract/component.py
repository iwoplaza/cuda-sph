

class Component:
    def draw(self, delta_time: float):
        pass

    def on_mouse_move(self, x: int, y: int) -> None:
        pass

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        return False

    def on_mouse_btn_released(self, x: int, y: int, button: int) -> None:
        pass

    def on_key_pressed(self, key: bytes) -> bool:
        return False

    def on_key_released(self, key: bytes) -> None:
        pass
