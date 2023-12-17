from transform import Transform
import numpy as np


class OrthoCamera:
    # The constructor takes six floats as arguments: left , right, bottom, top, near, and far.
    # These arguments define the orthographic projection of the camera used to construct ortho_transform.
    # The camera transform is initialized with the Transform default constructor.
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        self.ortho_transform = np.matrix([[(2 / (right - left)), 0, 0, -((right + left) / (right - left))],
                                          [0, (2 / (far - near)), 0, -((far + near) / (far - near))],
                                          [0, 0, (2 / (top - bottom)), -((top + bottom) / (top - bottom))],
                                          [0, 0, 0, 1]])

        # A Transform object exposed to set the orientation (position and rotation) of the camera.
        self.transform = Transform()

    # This method simply returns a float that is the ratio of the camera projection plane's width to height.
    # That is, if the screen width is 6 in world space and the screen height is 3, then this method would return 2.0.
    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)

    # This method takes a 3 element Numpy array, p, that represents a 3D point in world space as input.
    # It then transforms p to the camera coordinate system before performing an orthographic projection using the orthographic
    # transformation matrix and returns the resulting 3 element Numpy array that represents the point in screen space.
    def project_point(self, p):
        # convert to camera space
        transformed_point = self.transform.apply_inverse_to_point(p)
        # convert to screen space
        camera_space_point = np.append(np.transpose(np.asmatrix(transformed_point)), [[1]], axis=0)
        screen_space_point = np.matmul(self.ortho_transform, camera_space_point)
        temp = np.delete(screen_space_point, 3, 0)
        retVal = np.ravel(temp)
        return retVal

    # This method takes a 3 element Numpy array, p, that represents a 3D point in screen space as input.
    # It then transforms p to camera space before transforming back to world space using the inverse
    # orthographic transformation matrix and returns the resulting 3 element Numpy array.
    def inverse_project_point(self, p):
        # convert from screen space to camera space
        inv_ortho_mat = np.linalg.inv(self.ortho_transform)
        working_point = np.append(np.transpose(np.asmatrix(p)), [[1]], axis=0)
        temp0 = np.matmul(inv_ortho_mat, working_point)
        temp1 = np.delete(temp0, 3, 0)
        arrPoint = np.ravel(temp1)
        # convert from camera space to world space
        retVal = self.transform.apply_to_point(arrPoint)
        return retVal


class PerspectiveCamera:
    # The constructor takes six floats as arguments: left , right, bottom, top, near, and far.
    # These arguments define the orthographic projection of the camera used to construct the orthographic transformation.
    # The near and far values are also used to construct the perspective matrix.
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

        self.ortho_transform = np.matrix([[(2 / (right - left)), 0, 0, -((right + left) / (right - left))],
                                          [0, (2 / (far - near)), 0, -((far + near) / (far - near))],
                                          [0, 0, (2 / (top - bottom)), -((top + bottom) / (top - bottom))],
                                          [0, 0, 0, 1]])

        self.perspective_matrix = np.matrix([[near, 0, 0, 0],
                                             [0, near + far, 0, -(far * near)],
                                             [0, 0, near, 0],
                                             [0, 1, 0, 0]])

        self.ortho_transform_inv = np.linalg.inv(self.ortho_transform)
        self.perspective_matrix_inv = np.linalg.inv(self.perspective_matrix)

        # A Transform object exposed to set the orientation (position and rotation) of the camera. This should default to represent a position of (0, 0, 0) and no rotation.
        self.transform = Transform()

    # This method simply returns a float that is the ratio of the camera projection plane's width to height.
    # That is, if the screen width is 6 in world space and the screen height is 3, then this method would return 2.0.
    def ratio(self):
        return (self.right - self.left) / (self.top - self.bottom)

    # This method takes a 3 element Numpy array, p, that represents a 3D point in world space as input.
    # It then transforms p to the camera coordinate system before performing the perspective projection into screen space and returns the resulting 3 element Numpy array.
    def project_point(self, p):
        # convert to camera space
        transformed_point = self.transform.apply_inverse_to_point(p)
        # project point
        camera_space_point = np.append(np.transpose(np.asmatrix(transformed_point)), [[1]], axis=0)
        half_projected_point = np.matmul(self.perspective_matrix, camera_space_point)
        projected_point = half_projected_point / half_projected_point[3]
        # convert to screen space
        screen_space_point = np.matmul(self.ortho_transform, projected_point)
        temp = np.delete(screen_space_point, 3, 0)
        retVal = np.ravel(temp)
        return retVal

    # This method takes a 3 element Numpy array, p, that represents a 3D point in screen coordinates as input.
    # It then transforms p to the camera coordinate system before transforming back to world space and returns the resulting 3 element Numpy array.
    def inverse_project_point(self, p):
        # convert from screen space to camera space
        working_point = np.append(np.transpose(np.asmatrix(p)), [[1]], axis=0)
        projected_point = np.matmul(self.ortho_transform_inv, working_point)
        # un-project the point
        Yc = (self.far * self.near) / (self.near + self.far - projected_point[1])
        half_projected_point = projected_point * Yc
        camera_space_point = np.matmul(self.perspective_matrix_inv, half_projected_point)
        # convert from camera space to world space
        temp = np.delete(camera_space_point, 3, 0)
        arrPoint = np.ravel(temp)
        retVal = self.transform.apply_to_point(arrPoint)
        return retVal

    # fov is the camera's horizontal field of view in degrees, near is the distance to the near clipping plane,
    # far is the distance to the far clipping plane, and ratio is the pixel ratio of the final image (the value returned from screen.ratio()).
    # This static method will then compute left, right, top, and bottom to create a PerspectiveCamera object.
    # To do this, we will assume that left and right are symmetric, as are top and bottom. Finally, we will also assume that our pixels are square.
    @staticmethod
    def from_FOV(fov, near, far, ratio):
        right = near * np.tan((np.pi/180)*(fov/2))
        left = -1 * right
        top = ratio * left
        bottom = -1 * top
        camera = PerspectiveCamera(left, right, bottom, top, near, far)
        return camera