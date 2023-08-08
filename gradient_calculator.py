import numpy as np
from convolutor import Convolutor


class GradientCalculator:
    """
    Calculate the gradients using the sobel operator
    """

    def __init__(self, array):
        self.x_grad = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.y_grad = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.array = array

    def _get_x_grad(self):
        convolutor = Convolutor(self.array, self.x_grad)
        return convolutor.get_convolution()

    def _get_y_grad(self):
        convolutor = Convolutor(self.array, self.y_grad)
        return convolutor.get_convolution()

    def get_magnitudes_and_angles(self):
        grad_x = self._get_x_grad()
        grad_y = self._get_y_grad()
        r = len(grad_x)
        c = len(grad_x[0])
        assert r == len(grad_y) and c == len(grad_y[0])  # Ensure validity
        magnitudes = np.sqrt(np.square(grad_x) + np.square(grad_y))
        angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
        return magnitudes, angles
