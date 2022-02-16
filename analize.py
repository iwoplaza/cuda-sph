import numpy as np
from common.serializer.loader import Loader


if __name__ == '__main__':
    loader = Loader("simulation_out")
    params = loader.load_simulation_parameters()
    old_state = loader.load_simulation_state(0)
    for i in range(0, 200):
        new_state = loader.load_simulation_state(i)
        print(f"Epoch: {i}, Max values:")
        print("max position:", np.max(new_state.position))
        print("max velocity:", np.max(new_state.velocity))
        print("max density:", np.max(new_state.density))



