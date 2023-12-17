import numpy as np
from screen import Screen
from camera import PerspectiveCamera
from mesh import Mesh
# from renderers.sequential_renderer import Renderer
from renderers.renderer_shared_mem import Renderer
from light import PointLight

if __name__ == '__main__':
    screen = Screen(500, 500)

    camera = PerspectiveCamera(-1.0, 1.0, -1.0, 1.0, 1.0, 20)
    camera.transform.set_position(0, -4, 0)

    mesh_1 = Mesh.from_stl("../stl_files/suzanne.stl", np.array([1.0, 0.0, 1.0]), \
                           np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_1.transform.set_rotation(-15, 0, 215)
    mesh_1.transform.set_position(1, 1, 1)

    mesh_2 = Mesh.from_stl("../stl_files/unit_cube.stl", np.array([0.6, 0.0, 1.0]), \
                           np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_2.transform.set_position(-1, 0.5, -1)
    mesh_2.transform.set_rotation(10, -10, 0)

    mesh_3 = Mesh.from_stl("../stl_files/unit_sphere.stl", np.array([1.0, 0.6, 0.0]), \
                           np.array([1.0, 1.0, 1.0]), 0.05, 0.8, 0.2, 100)
    mesh_3.transform.set_position(-2, 0.25, 2)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(0, -5, 5)

    renderer = Renderer(screen, camera, [mesh_1, mesh_2, mesh_3], light)
    renderer.render("flat", [80, 80, 80], [0.2, 0.2, 0.2])

    screen.show()
