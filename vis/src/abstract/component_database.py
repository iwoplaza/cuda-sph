from typing import Dict
from .component import Component


class ComponentDatabase:
    registry: Dict[str, Component]

    def __init__(self):
        self.registry = {}

    def register(self, identifier: str, component: Component):
        self.registry[identifier] = component

    def get(self, identifier: str):
        return self.registry[identifier]
