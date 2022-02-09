from abc import ABC, abstractmethod

from common.main.data_classes.simulation_data_classes import SimulationState


class AbstractSPHStrategy(ABC):
    def __init__(self, params):
        self.params = params
        self.dt = 1 / self.params

    def compute_next_state(self, old_state) -> SimulationState:
        self._initialize_next_state(old_state)
        self._compute_density()
        self._compute_pressure()
        self._compute_viscosity()
        self._integrate()
        return self.next_state

    @property
    @abstractmethod
    def next_state(self) -> SimulationState:
        pass

    @abstractmethod
    def _initialize_next_state(self, old_state):
        pass

    @abstractmethod
    def _compute_density(self):
        pass

    @abstractmethod
    def _compute_pressure(self):
        pass

    @abstractmethod
    def _compute_viscosity(self):
        pass

    @abstractmethod
    def _integrate(self):
        pass
