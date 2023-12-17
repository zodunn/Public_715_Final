import numpy as np
import color
from threeDVector import ThreeDVector
from ray import Ray
import multiprocessing
from multiprocessing import shared_memory
import time


class Renderer:
    # The class constructor takes a screen object (of type Screen), camera object (either of type OrthoCamera or PerspectiveCamera),
    # a list of mesh objects (of type Mesh), and a light source (of type PointLight) and stores them.
    def __init__(self, screen, camera, meshes, light):
        self.screen = screen
        self.camera = camera
        self.meshes = meshes
        self.light = light

    # This method will take three input arguments.
    # shading is a string parameter indicating which type of shading to apply, and barycentric should be implemented.
    # bg_color is a three element list that is to be the background color of the render (that is, all of the pixels that are not part of a mesh).
    # ambient_light defines the intensity and color of any ambient lighting to add within the scene.
    # render will execute the basic render loop and compute shading at each pixel fragment to update an image buffer.
    # It will then draw that image buffer to the screen object using the screen.draw method, but it will not run the pygame loop (the calling function will call screen.show)
    def render(self, shading, bg_color, ambient_light):
        image_buffer = np.full((self.screen.width, self.screen.height, 3), bg_color)
        z_buffer = np.full((self.screen.width, self.screen.height), np.inf)
        # check camera and image buffer ratios
        if self.camera.ratio() != self.screen.ratio():
            exit(1)

        shared_z_buffer = shared_memory.SharedMemory(name='z_buffer', create=True, size=z_buffer.nbytes)
        shared_image_buffer = shared_memory.SharedMemory(name='image_buffer', create=True, size=image_buffer.nbytes)

        z_buf_shape = z_buffer.shape
        z_buf_type = z_buffer.dtype
        im_buf_shape = image_buffer.shape
        im_buf_type = image_buffer.dtype

        z_buffer_shared = np.ndarray(z_buf_shape, dtype=z_buf_type, buffer=shared_z_buffer.buf)
        z_buffer_shared[:] = z_buffer[:]
        image_buffer_shared = np.ndarray(im_buf_shape, dtype=im_buf_type, buffer=shared_image_buffer.buf)
        image_buffer_shared[:] = image_buffer[:]

        processes = []
        for pix_y in range(0, self.screen.height, 18):
            process = multiprocessing.Process(target=self.pixel_loop, args=(pix_y, pix_y+18, shading, ambient_light, z_buf_shape, z_buf_type, im_buf_shape, im_buf_type))
            processes.append(process)

        start_time = time.time()
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        end_time = time.time()
        self.screen.draw(image_buffer_shared)
        print(end_time - start_time)

        shared_z_buffer.close()
        shared_z_buffer.unlink()
        shared_image_buffer.close()
        shared_image_buffer.unlink()

    def pixel_loop(self, start_range, end_range, shading, ambient_light, z_buf_shape, z_buf_type, im_buf_shape, im_buf_type):
        existing_z_buffer = shared_memory.SharedMemory(name='z_buffer')
        existing_image_buffer = shared_memory.SharedMemory(name='image_buffer')
        z_buffer = np.ndarray(z_buf_shape, dtype=z_buf_type, buffer=existing_z_buffer.buf)
        image_buffer = np.ndarray(im_buf_shape, dtype=im_buf_type, buffer=existing_image_buffer.buf)
        for pix_y in range(start_range, end_range):
            for pix_x in range(0, self.screen.width):
                # create ray
                point_screen = self.screen.pixel_to_screen(pix_x, pix_y)
                point_screen[1] = -1
                point_world = self.camera.inverse_project_point(point_screen)
                direction = np.array([0, 1, 0])
                ray = Ray(point_world, None, direction)

                # for every triangle see if ray collides, if so color pixel
                for mesh in self.meshes:
                    for face in mesh.faces:
                        # convert vertices to world
                        world_vertices = []
                        for vertIndex in face:
                            world_vertices.append(mesh.transform.apply_to_point(mesh.verts[vertIndex]))

                        collides, t = ray.collide(world_vertices[0], world_vertices[1], world_vertices[2])
                        if collides:
                            if t > z_buffer[pix_x, pix_y]:
                                continue
                            z_buffer[pix_x, pix_y] = t

                            # calculate shadow, first get origin for new shadow ray
                            p = np.add(ray.e, np.multiply(ray.d, t))
                            # get normalized light vector
                            light_position = ThreeDVector.normalize(self.light.transform.apply_to_point(np.array([0, 0, 0])))
                            # move origin a small amount along light ray
                            shadow_origin = np.add(p, np.multiply(light_position, 0.01))
                            # create shadow ray
                            shadow_ray = Ray(shadow_origin, None, light_position)
                            # do shadow collision
                            shadow_collided = False
                            for shadow_mesh in self.meshes:
                                if self.meshes.index(mesh) == self.meshes.index(shadow_mesh):
                                    continue
                                for shadow_face in shadow_mesh.faces:
                                    # convert vertices to world
                                    shadow_world_vertices = []
                                    for shadow_vertIndex in shadow_face:
                                        shadow_world_vertices.append(shadow_mesh.transform.apply_to_point(shadow_mesh.verts[shadow_vertIndex]))

                                    shadow_collides, shadow_t = shadow_ray.collide(shadow_world_vertices[0], shadow_world_vertices[1], shadow_world_vertices[2])
                                    if shadow_collides:
                                        shadow_collided = True

                            if not shadow_collided:
                                if shading == 'flat':
                                    temp_face_normal = mesh.normals[mesh.faces.index(face)]
                                    face_normal = mesh.transform.apply_to_normal(temp_face_normal)
                                    final_color = color.ColorCalculation.flat(world_vertices, face_normal, mesh, self.light, ambient_light)
                                    display_color = color.ColorCalculation.calcFinalRGB(final_color)
                                if shading == 'phong-blinn':
                                    screen_vertices = []
                                    for vert in world_vertices:
                                        screen_vertices.append(self.camera.project_point(vert))
                                    a = screen_vertices[0]
                                    b = screen_vertices[1]
                                    c = screen_vertices[2]

                                    screen_x, screen_y, screen_z = self.screen.pixel_to_screen(pix_x, pix_y)
                                    gamma = ((a[2] - b[2]) * screen_x + (b[0] - a[0]) * screen_z + (a[0] * b[2]) - (b[0] * a[2])) / ((a[2] - b[2]) * c[0] + (b[0] - a[0]) * c[2] + (a[0] * b[2]) - (b[0] * a[2]))
                                    beta = ((a[2] - c[2]) * screen_x + (c[0] - a[0]) * screen_z + (a[0] * c[2]) - (c[0] * a[2])) / ((a[2] - c[2]) * b[0] + (c[0] - a[0]) * b[2] + (a[0] * c[2]) - (c[0] * a[2]))
                                    alpha = 1 - beta - gamma
                                    screen_y = alpha * a[1] + beta * b[1] + gamma * c[1]

                                    vertex_normal_1 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[0]])
                                    vertex_normal_2 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[1]])
                                    vertex_normal_3 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[2]])
                                    interpolated_normal_temp = np.add(np.add(np.multiply(vertex_normal_1, alpha), np.multiply(vertex_normal_2, beta)), np.multiply(vertex_normal_3, gamma))
                                    interpolated_normal = np.divide(interpolated_normal_temp, color.ColorCalculation.magnitude(interpolated_normal_temp))
                                    point_world = self.camera.inverse_project_point((screen_x, screen_y, screen_z))
                                    display_color = color.ColorCalculation.phong(point_world, interpolated_normal, mesh, self.light, ambient_light, self.camera.transform.apply_to_point(np.array([0, 0, 0])))
                            else:
                                display_color = color.ColorCalculation.calcFinalRGB(np.multiply(ambient_light, mesh.ka))

                            image_buffer[pix_x, pix_y] = display_color
        existing_z_buffer.close()
        existing_image_buffer.close()
