import numpy as np
from screen import Screen
from camera import PerspectiveCamera
from mesh import Mesh
# from renderers.sequential_renderer import Renderer
from renderers.renderer_shared_mem import Renderer
from light import PointLight

if __name__ == '__main__':
    screen = Screen(500, 500)

    camera = PerspectiveCamera(-1.0, 1.0, -1.0, 1.0, 1.0, 10)
    camera.transform.set_position(0, -3, 0)

    mesh = Mesh.from_stl("../stl_files/suzanne.stl", np.array([1.0, 0.0, 1.0]), \
                         np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh.transform.set_rotation(-15, 0, 200)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(0, -5, 5)

    renderer = Renderer(screen, camera, [mesh], light)
    renderer.render("flat", [80, 80, 80], [0.2, 0.2, 0.2])

    screen.show()
