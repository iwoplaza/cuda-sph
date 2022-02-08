from abc import ABC, abstractmethod

from common.main.data_classes.simulation_data_classes import SimulationState

class SPHStrategy(ABC):
    def __init__(self, parameters):
        self.__parameters = parameters

    @property 
    @abstractmethod 
    def next_state(self) -> SimulationState:
        pass 

    def compute_next_state(self, old_state) -> SimulationState:
        self.__initialize_next_state(old_state)
        self.__compute_density()
        self.__compute_pressure()
        self.__compute_viscosity()
        self.__integrate()
        return self.next_state
        
    @abstractmethod
    def __initialize_next_state(self):
        pass

    @abstractmethod
    def __compute_density(self):
        pass 

    @abstractmethod
    def __compute_pressure(self):
        pass 
    
    @abstractmethod
    def __compute_viscosity(self):
        pass 

    @abstractmethod
    def __integrate(self):
        pass 


