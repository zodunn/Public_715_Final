import numpy as np
from screen import Screen
from camera import OrthoCamera
from mesh import Mesh
# from renderers.renderer_ray_traced_sequential import Renderer
from renderers.renderer_ray_traced import Renderer
from light import PointLight

if __name__ == '__main__':
    screen = Screen(500, 500)

    camera = OrthoCamera(-1.5, 1.5, -1.5, 1.5, 1.0, 10)
    camera.transform.set_position(0, -8, 0)

    mesh_1 = Mesh.from_stl("../stl_files/unit_cube.stl", np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_1.transform.set_rotation(5, 0, -10)
    mesh_1.transform.set_position(0.8, -1.5, 0)

    mesh_2 = Mesh.from_stl("../stl_files/unit_cube.stl", np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_2.transform.set_rotation(5, 0, -10)
    mesh_2.transform.set_position(-0.8, -1.5, 0)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(3, -2, 0)

    renderer = Renderer(screen, camera, [mesh_1, mesh_2], light)
    renderer.render("flat", [80, 80, 80], [0.2, 0.2, 0.2])

    screen.show()