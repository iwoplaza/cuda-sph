from ..vector import Vec3f
from vis.src.abstract.component_database import ComponentDatabase
from vis.src.abstract.scene_components import Camera


class Command:
    type: str

    def __init__(self, cmd_type: str):
        self.type = cmd_type

    def execute(self, component_database: ComponentDatabase):
        raise NotImplementedError()


class PositionCamera(Command):
    camera_id: str
    position: Vec3f
    pitch: float = None
    yaw: float = None

    def __init__(self, camera_id: str, position: Vec3f, pitch: float = None, yaw: float = None):
        super().__init__('position-camera')

        self.camera_id = camera_id
        self.position = position
        self.pitch = pitch
        self.yaw = yaw

    def execute(self, component_database: ComponentDatabase):
        camera: Camera = component_database.get(self.camera_id)

        camera.set_position(self.position)
        if self.yaw is not None:
            camera.set_yaw(self.yaw)
        if self.pitch is not None:
            camera.set_pitch(self.pitch)
