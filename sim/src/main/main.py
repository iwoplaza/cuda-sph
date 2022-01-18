from __future__ import annotations
from utils import SimulationState
from state_generator import StateGenerator


class Serializer:
    def serialize(self, state: SimulationState) -> None:
        print(f"Serialized < {state} >")


def main():
    serializer = Serializer()
    for frame in StateGenerator(None):
        serializer.serialize(frame)


if __name__ == '__main__':
    main()