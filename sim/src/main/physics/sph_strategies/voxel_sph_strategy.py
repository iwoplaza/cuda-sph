from common.main.data_classes.simulation_data_classes import SimulationState
from sim.src.main.physics.sph_strategies.abstract_sph_strategy import AbstractSPHStrategy


# TODO


class VoxelSPHStrategy(AbstractSPHStrategy):
    @property
    def next_state(self) -> SimulationState:
        return SimulationState(None, None, None, None)

    def _initialize_next_state(self, old_state):
        pass

    def _compute_density(self):
        pass

    def _compute_pressure(self):
        pass

    def _compute_viscosity(self):
        pass

    def _integrate(self):
        pass
