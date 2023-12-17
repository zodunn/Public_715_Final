import numpy as np
import math


class ColorCalculation:
    @staticmethod
    def flat(face_vertices, face_normal, mesh, light, ambient_color):
        # calculate intensity E = I * color * 1/d^2 * cos(theta)
        I = light.intensity
        color = light.color
        light_position = light.transform.apply_to_point(np.array([0, 0, 0]))
        point_world = np.add(np.add(np.divide(face_vertices[0], 3), np.divide(face_vertices[1], 3)), np.divide(face_vertices[2], 3))
        l = np.subtract(light_position, point_world)
        d = ColorCalculation.magnitude(l)
        cos_theta = max(0.0, np.dot(l, face_normal) / d)
        E = I * color * (1/pow(d, 2)) * cos_theta

        # calculate the diffuse shading component Cd/pi * Kd
        diffuse = (mesh.kd/math.pi) * mesh.diffuse_color

        specular = np.array([0, 0, 0])

        # final Lambertian reflectance
        Lr = np.add(np.multiply(np.add(diffuse, specular), E), np.multiply(ambient_color, mesh.ka))

        return Lr

    @staticmethod
    def barycentric(alpha, beta, gamma):
        color = alpha * np.array([255, 0, 0]) + beta * np.array([0, 255, 0]) + gamma * np.array([0, 0, 255])
        return np.array([int(color[0]), int(color[1]), int(color[2])])

    @staticmethod
    def magnitude(vector):
        return math.sqrt(sum(pow(element, 2) for element in vector))

    @staticmethod
    def calcFinalRGB(color):
        return [min(255, int(color[0] * 255)), min(255, int(color[1] * 255)), min(255, int(color[2] * 255))]

    @staticmethod
    def getMinMaxDepth(meshes, camera):
        min_depth = np.inf
        max_depth = -np.inf
        all_vertices = []
        for mesh in meshes:
            for face in mesh.faces:
                # Normal culling
                temp_face_normal = mesh.normals[mesh.faces.index(face)]
                face_normal = mesh.transform.apply_to_normal(temp_face_normal)
                camera_normal = camera.transform.apply_to_normal(np.array([0, 1, 0]))
                if np.dot(face_normal, camera_normal) >= 0:
                    continue

                # convert vertices to world space
                world_vertices = []
                for vertIndex in face:
                    world_vertices.append(mesh.transform.apply_to_point(mesh.verts[vertIndex]))

                # convert world vertices to screen space
                screen_vertices = []
                for vert in world_vertices:
                    all_vertices.append(camera.project_point(vert))
        for vert in all_vertices:
            if vert[1] > max_depth:
                max_depth = vert[1]
            if vert[1] < min_depth:
                min_depth = vert[1]
        return min_depth, max_depth

    @staticmethod
    def depth(min_depth, max_depth, screen_y):
        black = np.array([0, 0, 0])
        white = np.array([255, 255, 255])
        if screen_y == min_depth:
            color = black
        elif screen_y == max_depth:
            color = white
        else:
            total_distance = max_depth - min_depth
            distance_from_min = screen_y - min_depth
            percent = distance_from_min/total_distance
            c = (black * (1 - percent)) + (white * percent)
            color = np.array([int(c[0]), int(c[1]), int(c[2])])
        return color

    @staticmethod
    def phong(point_world, interpolated_normal, mesh, light, ambient_color, camera_pos):
        # calculate intensity E = I * color * 1/d^2 * cos(theta)
        I = light.intensity
        color = light.color
        light_position = light.transform.apply_to_point(np.array([0, 0, 0]))
        l = np.subtract(light_position, point_world)
        d = ColorCalculation.magnitude(l)
        l = l / d
        cos_theta = max(0.0, np.dot(l, interpolated_normal))
        E = I * color * (1/pow(d, 2)) * cos_theta

        # calculate the diffuse shading component Cd/pi * Kd
        diffuse = (mesh.kd/math.pi) * mesh.diffuse_color

        # calculate the specular shading component
        v = np.subtract(camera_pos, point_world)
        v = v / ColorCalculation.magnitude(v)
        h = np.add(l, v)
        h_norm = np.divide(h, ColorCalculation.magnitude(h))
        specular = mesh.ks * mesh.specular_color * pow(max(0, np.dot(h_norm, interpolated_normal)), mesh.ke)

        # final Lambertian reflectance
        Lr = np.add(np.multiply(np.add(diffuse, specular), E), np.multiply(ambient_color, mesh.ka))

        return ColorCalculation.calcFinalRGB(Lr)

    @staticmethod
    def gouraud(point_world, vertex_normal, mesh, light, ambient_color, camera_pos):
        # calculate intensity E = I * color * 1/d^2 * cos(theta)
        I = light.intensity
        color = light.color
        light_position = light.transform.apply_to_point(np.array([0, 0, 0]))
        l = np.subtract(light_position, point_world)
        d = ColorCalculation.magnitude(l)
        l = l / d
        cos_theta = max(0.0, np.dot(l, vertex_normal))
        E = I * color * (1 / pow(d, 2)) * cos_theta

        # calculate the diffuse shading component Cd/pi * Kd
        diffuse = (mesh.kd / math.pi) * mesh.diffuse_color

        # calculate the specular shading component
        v = np.subtract(camera_pos, point_world)
        v = v / ColorCalculation.magnitude(v)
        h = np.add(l, v)
        h_norm = np.divide(h, ColorCalculation.magnitude(h))
        specular = mesh.ks * mesh.specular_color * pow(max(0, np.dot(h_norm, vertex_normal)), mesh.ke)

        # final Lambertian reflectance
        Lr = np.add(np.multiply(np.add(diffuse, specular), E), np.multiply(ambient_color, mesh.ka))

        return ColorCalculation.calcFinalRGB(Lr)