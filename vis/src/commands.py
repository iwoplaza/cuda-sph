from dataclasses import dataclass
from .vector import Vec3f


class Command:
    type: str


@dataclass
class PositionCamera(Command):
    type = 'position-camera'
    position: Vec3f
    pitch: float = None
    yaw: float = None
