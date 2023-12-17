from stl import mesh as numpyMesh
import threeDVector as v
from transform import Transform


class Mesh:
    # The constructor takes diffuse and specular color as an 3 element np array with all three values between 0.0 and 1.0, as well as material properties ka, kd, ks, and ke.
    def __init__(self, diffuse_color, specular_color, ka, kd, ks, ke):
        # List of 3D vertices <x,y,z> for the mesh.
        self.verts = None
        # List of triangle faces for the mesh, with each face defined as a list of 3 vertex indices into verts in counterclockwise ordering.
        self.faces = None
        # List of 3D face normals for the mesh. The elements of this list correspond to the same triangles defined in faces.
        self.normals = None
        # List of vertex normals for the mesh. The elements of the list correspond to the same vertices defined in verts.
        self.vertex_normals = None
        # Transform object member
        self.transform = Transform()

        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke

    # This static method takes an stl file as input, initializes an empty Mesh object using the input material properties diffuse_color,
    # specular_color, ka, kd, ks, ke and populates the verts, faces, and normals member variables. The method returns the populated Mesh object.
    @staticmethod
    def from_stl(stl_path, diffuse_color, specular_color, ka, kd, ks, ke):
        # get mesh
        meshInput = numpyMesh.Mesh.from_file(stl_path)
        # initialize mesh object
        mesh = Mesh(diffuse_color, specular_color, ka, kd, ks, ke)
        mesh.verts = []
        mesh.faces = []
        mesh.normals = []
        mesh.vertex_normals = []

        # for each triangle in the mesh parse out its vertices and store the faces and vertices in two separate lists
        for triangle in meshInput:
            facePoints = []
            for i in range(0, len(triangle), 3):
                vertex = tuple(triangle[i:i+3])  # grab numbers from the triangle 3 at a time, so you get one of the vertices of the triangle
                if vertex not in mesh.verts:  # only add a vertex to the list once, if we encounter a vertex from a triangle that has already been added, ignore it
                    mesh.verts.append(vertex)
                facePoints.append(mesh.verts.index(vertex))
            mesh.faces.append(tuple(facePoints))  # once we have looped through the whole triangle record the vertices that make up its face as indices into the vertex list

        # calculate and store the normals for each of the triangles
        for face in mesh.faces:
            mesh.normals.append(v.ThreeDVector.find_normal([mesh.verts[face[0]], mesh.verts[face[1]], mesh.verts[face[2]]]))

        # collect all the face normals for the faces that are touching a vertex for all vertices
        face_normals_for_vertices = [ [] for _ in range(len(mesh.verts))]
        for face in mesh.faces:
            index_of_face = mesh.faces.index(face)
            face_normals_for_vertices[face[0]].append(mesh.normals[index_of_face])
            face_normals_for_vertices[face[1]].append(mesh.normals[index_of_face])
            face_normals_for_vertices[face[2]].append(mesh.normals[index_of_face])

        # calculate and store the normals for the vertices
        for i in range(0, len(mesh.verts)):
            mesh.vertex_normals.append(v.ThreeDVector.vertex_normal(face_normals_for_vertices[i]))

        return mesh