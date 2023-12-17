import numpy as np
from color import ColorCalculation
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

        start_time = time.time()
        # depth shader
        if shading == 'depth':
            min_depth, max_depth = ColorCalculation.getMinMaxDepth(self.meshes, self.camera)

        for mesh in self.meshes:
            for face in mesh.faces:
                # Normal culling
                temp_face_normal = mesh.normals[mesh.faces.index(face)]
                face_normal = mesh.transform.apply_to_normal(temp_face_normal)
                camera_normal = self.camera.transform.apply_to_normal(np.array([0, 1, 0]))
                if np.dot(face_normal, camera_normal) >= 0:
                    continue

                # convert vertices to world space
                world_vertices = []
                for vertIndex in face:
                    world_vertices.append(mesh.transform.apply_to_point(mesh.verts[vertIndex]))

                # convert world vertices to screen space
                screen_vertices = []
                for vert in world_vertices:
                    screen_vertices.append(self.camera.project_point(vert))

                # convert to pixel space
                pixel_vertices = []
                for vert in screen_vertices:
                    pixel_vertices.append(self.screen.screen_to_pixel(vert))

                # A - pre-fragment shading calculations
                if shading == 'flat':
                    final_color = ColorCalculation.flat(world_vertices, face_normal, mesh, self.light, ambient_light)
                if shading == 'gouraud':
                    vertex_colors = [ColorCalculation.gouraud(world_vertices[0], mesh.transform.apply_to_normal(mesh.vertex_normals[face[0]]), mesh, self.light, ambient_light, self.camera.transform.apply_to_point(np.array([0, 0, 0]))),
                                     ColorCalculation.gouraud(world_vertices[1], mesh.transform.apply_to_normal(mesh.vertex_normals[face[1]]), mesh, self.light, ambient_light, self.camera.transform.apply_to_point(np.array([0, 0, 0]))),
                                     ColorCalculation.gouraud(world_vertices[2], mesh.transform.apply_to_normal(mesh.vertex_normals[face[2]]), mesh, self.light, ambient_light, self.camera.transform.apply_to_point(np.array([0, 0, 0])))]

                # find rectangular bounds of face (in pixel space)
                min_x = self.screen.width
                min_y = self.screen.height
                max_x = 0
                max_y = 0
                for vert in pixel_vertices:
                    if vert[0] > max_x:
                        max_x = vert[0]
                    if vert[0] < min_x:
                        min_x = vert[0]
                    if vert[1] > max_y:
                        max_y = vert[1]
                    if vert[1] < min_y:
                        min_y = vert[1]

                # for valid pixel in bounds
                a = screen_vertices[0]
                b = screen_vertices[1]
                c = screen_vertices[2]

                for pix_y in range(min_y, max_y + 1):
                    for pix_x in range(min_x, max_x + 1):
                        if pix_y >= self.screen.height or pix_x >= self.screen.width:
                            continue
                        screen_x, screen_y, screen_z = self.screen.pixel_to_screen(pix_x, pix_y)
                        gamma = ((a[2] - b[2]) * screen_x + (b[0] - a[0]) * screen_z + (a[0] * b[2]) - (b[0] * a[2])) / ((a[2] - b[2]) * c[0] + (b[0] - a[0]) * c[2] + (a[0] * b[2]) - (b[0] * a[2]))
                        beta = ((a[2] - c[2]) * screen_x + (c[0] - a[0]) * screen_z + (a[0] * c[2]) - (c[0] * a[2])) / ((a[2] - c[2]) * b[0] + (c[0] - a[0]) * b[2] + (a[0] * c[2]) - (c[0] * a[2]))
                        alpha = 1 - beta - gamma
                        screen_y = alpha * a[1] + beta * b[1] + gamma * c[1]
                        if screen_y > 1 or screen_y < -1:
                            continue
                        if screen_y > z_buffer[pix_x, pix_y]:
                            continue
                        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1):
                            z_buffer[pix_x, pix_y] = screen_y
                            # B - calculate fragment color -> color
                            if shading == 'flat':
                                display_color = ColorCalculation.calcFinalRGB(final_color)
                            elif shading == 'barycentric':
                                display_color = ColorCalculation.barycentric(alpha, beta, gamma)
                            elif shading == 'depth':
                                display_color = ColorCalculation.depth(min_depth, max_depth, screen_y)
                            elif shading == 'phong-blinn':
                                vertex_normal_1 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[0]])
                                vertex_normal_2 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[1]])
                                vertex_normal_3 = mesh.transform.apply_to_normal(mesh.vertex_normals[face[2]])
                                interpolated_normal_temp = np.add(np.add(np.multiply(vertex_normal_1, alpha), np.multiply(vertex_normal_2, beta)), np.multiply(vertex_normal_3, gamma))
                                interpolated_normal = np.divide(interpolated_normal_temp, ColorCalculation.magnitude(interpolated_normal_temp))
                                point_world = self.camera.inverse_project_point((screen_x, screen_y, screen_z))
                                display_color = ColorCalculation.phong(point_world, interpolated_normal, mesh, self.light, ambient_light, self.camera.transform.apply_to_point(np.array([0, 0, 0])))
                            elif shading == 'gouraud':
                                display_color = np.add(np.add(np.multiply(vertex_colors[0], alpha), np.multiply(vertex_colors[1], beta)), np.multiply(vertex_colors[2], gamma))
                                display_color[0] = int(display_color[0])
                                display_color[1] = int(display_color[1])
                                display_color[2] = int(display_color[2])

                            image_buffer[pix_x, pix_y] = display_color

        end_time = time.time()
        self.screen.draw(image_buffer)
        print(end_time - start_time)
