# About
The CUDA-SPH project aims to provide a set of tools for simulating fluids, as well as for visualizing the generated data.
To simulate fluid we used the [SPH technique](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics).
This approach assumes that a fluid is a set of particles that interact with each other and move around according to [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier–Stokes_equations).

We saw an opportunity to parallelize it, so we used GPU programming with [CUDA](https://developer.nvidia.com/cuda-toolkit) to implement it.
To make production relatively faster and easier for us, we used the Python programming language.

# Setup
Tested with Python 3.9

## Simulator workspace
Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
```sh
cd sim
./venv/Scripts/activate
pip install -r ./requirements.txt
```

## Visualization workspace
Download PyOpenGL manualy from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl.
```sh
cd vis
./venv/Scripts/activate
pip install ./PyOpenGL-3.1.5-cp39-cp39-win_amd64.whl # cp39 <==> Python 3.9
pip install ./PyOpenGL_accelerate-3.1.5-cp39-cp39-win_amd64.whl # cp39 <==> Python 3.9
pip install -r ./requirements.txt
```

# Usage

Software we made is divided into two programs, simulator and visualizer.
Simulator is able to produce a simulation (to a chosen directory), and visualizer displays it.


## Simulator
Two types of enclosure are available:
- inside a box
- inside a pipe

The `config.py` file is used to set numerous simulation parameters such as the mode (enclosure type), particle count and the initial state of the simulation.

Running the simulator:
```shell
python ./sim/src/main.py
```

## Visualizer
Running the visualizer:
```shell
python ./vis/src/main.py
```

### Movement
- `W,A,S,D` - Move forward, left, back and right (relative to the camera)
- `Q, E` - Move down and up (relative to the camera)
- `Space` - Increase the speed of moving around twofold
- `Click and drag` - Rotate the camera

### Demo

![Box 1](docs/screens/box1.png)
![Box 2](docs/screens/box2.png)
![Pipe](docs/screens/pipe.png)

# Read more
Full documentation of the project is available in Polish: [Dokumentacja Projektu](docs/Dokumentacja%20projektu%20CUDA-SPH.pdf)