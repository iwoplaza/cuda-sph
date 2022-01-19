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
# Development
## Tasks:
- Common
    - Zapis/odczyt danych o symulacji z pliku (Parametry symulacji, stan względny od czasu) [Paweł]
    - Definicja rurki: { origin: vec3, radius: float } [Paweł/Iwo]
- Symulacja
    - DONE Szkietet programu [Maciej/Tomek] 
    - DONE Kernele (przynajmniej zarys) [Maciej/Tomek]
    - Optymalizacja przeszukiwania n^2 za pomocą sąsiadów/voxeli
    - podłazcenie do serializera (na razie korzysta ze swojego, mockowego)
- Wizualizer [Iwo]
    - Szkielet projektu [Iwo]
    - Ruch kamery.
    - Shader managment.
    - GUI (imgui?)
