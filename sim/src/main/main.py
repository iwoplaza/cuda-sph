from __future__ import annotations

import numpy as np
from simulation_data import Pipe, SimulationParameters, SimulationState
from state_generator import StateGenerator


class Serializer:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = open(filename, "w")

    def serialize(self, state: SimulationState) -> None:
        self.file.write(str(state.position[0]) + '\n')
    
    def close(self):
        self.file.close()


if __name__ == '__main__':

    params = SimulationParameters(
        n_particles=100,
        external_force=np.asarray([0.5, 0.0, 0.0], dtype=np.float64),
        simulation_duration=1,
        fps=30,
        pipe=Pipe(segments=None),
        space_dims=(1, 1, 1),
        voxel_dim=(1e-4, 1e-4, 1e-4)
    )

    start_state = SimulationState(
        position=np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64"),
        velocity=np.random.random(params.n_particles * 3).reshape((params.n_particles, 3)).astype("float64"),
        density=np.zeros(params.n_particles).astype("float64"),
        voxel=np.zeros(params.n_particles).astype("int32")
    )

    serializer = Serializer("results.txt")
    for frame in StateGenerator(start_state, params):
        serializer.serialize(frame)
    serializer.close()
