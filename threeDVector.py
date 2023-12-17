import numpy as np
from math import sqrt, pow


class ThreeDVector:
    # return the cross product of two numpy arrays which are vectors and cast it to a tuple
    @staticmethod
    def cross(tuple1, tuple2):
        return tuple(np.cross(tuple1, tuple2))

    # return the difference of two vertices to get a vector, returned as a np array
    @staticmethod
    def subtract(tuple1, tuple2):
        return np.add(list(tuple1), np.negative(list(tuple2)))

    # normalize the vector so that it is of distance 1 and return a tuple
    @staticmethod
    def normalize(tuple1):
        mag = sqrt(pow(tuple1[0], 2) + pow(tuple1[1], 2) + pow(tuple1[2], 2))

        return tuple1[0] / mag, tuple1[1] / mag, tuple1[2] / mag

    # return the normal of two vectors, retval will be a tuple since that is what the cross method returns
    @staticmethod
    def find_normal(face):
        vector1 = ThreeDVector.subtract(face[1], face[0])
        vector2 = ThreeDVector.subtract(face[2], face[0])

        return ThreeDVector.normalize(ThreeDVector.cross(vector1, vector2))

    @staticmethod
    def vertex_normal(face_normals):
        result = (0, 0, 0)
        for norm in face_normals:
            result = np.add(result, norm)
        retVal = ThreeDVector.normalize(result)
        return retVal