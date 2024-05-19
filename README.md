
# Ray Tracing Project

## Overview
This project implements a basic ray tracing engine in Python. It supports rendering of simple 3D scenes consisting of various geometric shapes and supports different lighting models. The project is structured to demonstrate the capabilities of ray tracing in generating realistic images by simulating the interactions of light with objects.

## Project Structure
- `hw3.py`: The main script which handles the rendering process. It uses ray tracing algorithms to compute the color of each pixel based on light interactions.
- `helper_classes.py`: Contains helper classes and functions such as light sources and vector operations which are essential for ray tracing calculations.
- `Ray_Tracing_Assignment.ipynb`: A Jupyter notebook that provides an interactive environment to explore ray tracing concepts and see immediate results of the code.
- `scenes/`: Directory containing data files used to define various scenes. These might include object definitions, lighting setups, and camera configurations.
- Scene images (`scene1.png`, `scene2.png`, etc.): Output images from rendered scenes.

## Detailed Analysis
The `helper_classes.py` contains essential definitions for light sources and utility functions for vector operations. The `hw3.py` script is responsible for setting up the scene, including camera, objects, and lights, and uses ray tracing algorithms to render the scene.

## Setup and Running the Project
To run this project, ensure you have Python installed along with the necessary libraries:

```bash
pip install numpy matplotlib
```

To render a scene, run the `hw3.py` script:

```bash
python hw3.py
```

The script will produce an image file as output, showing the rendered scene based on the predefined setup in the code.

## License
This project is open-source and available for personal and educational use.

## Contact
For any further questions or contributions to the project, please submit an issue or pull request on GitHub.
