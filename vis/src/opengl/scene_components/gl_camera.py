import glm
import numpy as np

from vis.src.abstract.scene_components import Camera
from vis.src.vector import Vec3f


class GLCamera(Camera):
    def __init__(self, window, origin: Vec3f, yaw: float = None, pitch: float = None):
        super().__init__(origin=origin)

        self.__window = window
        self.proj_mat = None
        self.view_mat = glm.mat4(1)
        self.position = glm.vec3(*origin)

        self.__dragging = False
        self.__last_mouse_coords = None
        self.__move_direction = glm.vec3(0, 0, 0)

        self.__pressed = dict()
        self.__pitch = pitch
        self.__yaw = yaw

        self.__update_view()
        self.enable()

    def __update_view(self):
        translation = glm.translate(glm.mat4(1), -self.position)
        rotation = glm.rotate(glm.mat4(1), self.__pitch, glm.vec3(1, 0, 0))
        rotation = glm.rotate(rotation, self.__yaw, glm.vec3(0, 1, 0))
        self.view_mat = rotation * translation

    def __is_key_pressed(self, key: str):
        return self.__pressed[key] if key in self.__pressed else False

    def __calculate_target_move_direction(self):
        vec = glm.vec3(0, 0, 0)
        if self.__is_key_pressed('a'):
            vec.x -= 1
        if self.__is_key_pressed('d'):
            vec.x += 1

        if self.__is_key_pressed('w'):
            vec.z -= 1
        if self.__is_key_pressed('s'):
            vec.z += 1

        if self.__is_key_pressed('q'):
            vec.y -= 1
        if self.__is_key_pressed('e'):
            vec.y += 1

        if glm.length2(vec) > 1:
            return glm.normalize(vec)

        return vec

    def setup_projection(self, width: int, height: int):
        self.proj_mat = glm.perspective(np.pi / 2, width / height, 0.01, 1000)

    def enable(self):
        self.__window.use_camera(self)

    def on_key_pressed(self, key: bytes) -> bool:
        try:
            decoded = key.decode("utf-8").lower()
            self.__pressed[decoded] = True
        except UnicodeDecodeError:
            pass

        return True

    def on_key_released(self, key: bytes) -> None:
        try:
            decoded = key.decode("utf-8").lower()
            self.__pressed[decoded] = False
        except UnicodeDecodeError:
            pass

    def on_mouse_btn_pressed(self, x: int, y: int, button: int) -> bool:
        self.__last_mouse_coords = (x, y)
        return True

    def on_mouse_move(self, x: int, y: int) -> None:
        if self.__last_mouse_coords is None:
            return

        lx, ly = self.__last_mouse_coords
        dx, dy = x - lx, y - ly
        self.__last_mouse_coords = (x, y)

        sensitivity = 0.01
        self.__yaw += dx * sensitivity
        self.__pitch = min(max(-np.pi / 2, self.__pitch + dy * sensitivity), np.pi / 2)

        self.__update_view()

    def on_mouse_btn_released(self, x: int, y: int, button: int) -> None:
        self.__last_mouse_coords = None

    def set_position(self, position: Vec3f):
        self.position.x = position[0]
        self.position.y = position[1]
        self.position.z = position[2]
        self.__update_view()

    def set_yaw(self, yaw: float):
        self.__yaw = yaw
        self.__update_view()

    def set_pitch(self, pitch: float):
        self.__pitch = pitch
        self.__update_view()

    def update(self, dt: float):
        self.__move_direction = self.__calculate_target_move_direction()

        move = glm.vec4(self.__move_direction.x, self.__move_direction.y, self.__move_direction.z, 0)
        # Moving from the perspective of the camera
        move = glm.inverse(self.view_mat) * move

        move_speed = 10
        move *= move_speed

        self.position = self.position + glm.vec3(move * dt)
        self.__update_view()

