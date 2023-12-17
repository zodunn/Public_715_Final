import numpy as np


class Transform:
    # The constructor takes no required arguments.
    def __init__(self):
        # Matrix to store transform
        self.transformMatrix = np.identity(4)

    # Returns a 4x4 Numpy matrix that represents the transformation matrix
    def transformation_matrix(self):
        return self.transformMatrix

    # This method takes three scalars (x, y, and z) as input, and updates the Transform object's internal state to represent a new position at (x, y, z).
    def set_position(self, x, y, z):
        self.transformMatrix[0, 3] = x
        self.transformMatrix[1, 3] = y
        self.transformMatrix[2, 3] = z

    # This method takes three scalars (x, y, and z) as input, and updates the Transform object's internal rotation state.
    # The input values x, y, and z are expected to be degrees values between 0.0 and 360.0 (there is no need to check this),
    # and represent the amount of rotation around the x, y, and z axis respectively. The rotation should be set using the XYZ order of rotation.
    def set_rotation(self, x, y, z):
        x_rot = np.matrix([[1, 0, 0], [0, np.cos(x * (np.pi/180)), -1 * np.sin(x * (np.pi/180))], [0, np.sin(x * (np.pi/180)), np.cos(x * (np.pi/180))]])
        y_rot = np.matrix([[np.cos(y * (np.pi/180)), 0, np.sin(y * (np.pi/180))], [0, 1, 0], [-1 * np.sin(y * (np.pi/180)), 0, np.cos(y * (np.pi/180))]])
        z_rot = np.matrix([[np.cos(z * (np.pi/180)), -1 * np.sin(z * (np.pi/180)), 0], [np.sin(z * (np.pi/180)), np.cos(z * (np.pi/180)), 0], [0, 0, 1]])
        rotMat = np.matmul((np.matmul(x_rot, y_rot)), z_rot)

        for row in range(0, 3):
            for col in range(0, 3):
                self.transformMatrix[row, col] = rotMat[row, col]

    # This method returns a 4x4 Numpy matrix that is the inverse of the transformation matrix.
    def inverse_matrix(self):
        # get matrix parts:
        A = self.transformMatrix[0:3:1, 0:3:1]
        t = self.transformMatrix[0:3:1, 3]
        # transpose them
        A_transpose = np.transpose(A)
        new_t = np.matmul((-1 * A_transpose), t)
        # re-assemble matrix
        retVal = np.zeros((4, 4))
        retVal[3, 3] = 1
        for row in range(0, 3):
            for col in range(0, 3):
                retVal[row, col] = A_transpose[row, col]
        for row in range(0, 3):
            retVal[row, 3] = new_t[row]
        return retVal

    # This method takes a 3 element Numpy array, p, that represents a 3D point in space as input. It then applies the transformation matrix to it, and returns the resulting 3 element Numpy array.
    def apply_to_point(self, p):
        point = np.transpose(np.concatenate((np.matrix(p), [[1]]), 1))
        result = np.matmul(self.transformMatrix, point)
        retVal = np.transpose(np.delete(result, 3, 0))
        return np.ravel(retVal)

    # This method takes a 3 element Numpy array, p, that represents a 3D point in space as input. It then applies the inverse transformation matrix to it, and returns the resulting 3 element Numpy array.
    def apply_inverse_to_point(self, p):
        point = np.transpose(np.concatenate((np.matrix(p), [[1]]), 1))
        result = np.matmul(self.inverse_matrix(), point)
        retVal = np.transpose(np.delete(result, 3, 0))
        return np.ravel(retVal)

    # This method takes a 3 element Numpy array, n, that represents a 3D normal vector. It then applies the transform's rotation to it,
    # and returns the resulting 3 element Numpy array. The resulting array should be normalized and should not be affected by any positional component within the transform.
    def apply_to_normal(self, n):
        point = np.transpose(np.concatenate((np.matrix(n), [[0]]), 1))
        result = np.matmul(self.transformMatrix, point)
        retVal = np.transpose(np.delete(result, 3, 0))
        return np.ravel(retVal)

    # This method will take a 3D numpy array as the axis input, and a scalar in degrees as the rotation input.
    # This method will calculate the rotation resulting from the amount of input rotation rotating around the
    # input axis and update its internal rotation representation as such
    def set_axis_rotation(self, axis, rotation):
        # get rotation part of transformation matrix
        A = self.transformMatrix[0:3:1, 0:3:1]
        # calculate rotation using Rodrigues' rotation formula:
        # v_rot = (cos(theta)v + sin(theta)(v cross e) + (1 - cos(theta))(v dot e)v)
        temp1 = np.cos(rotation * (np.pi/180)) * A
        temp2 = np.sin(rotation * (np.pi/180)) * (np.cross(A, axis))
        temp3 = (1 - np.cos(rotation * (np.pi/180))) * (np.dot(A, axis) * A)
        v_rot = temp1 + temp2 + temp3

        for row in range(0, 3):
            for col in range(0, 3):
                self.transformMatrix[row, col] = v_rot[row, col]