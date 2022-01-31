from common.main.data_classes.simulation_data_classes import SimulationState


class Serializer:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = open(filename, "w")

    def serialize(self, state: SimulationState) -> None:
        self.file.write(str(state.position[0]) + '\n')
    
    def close(self):
        self.file.close()