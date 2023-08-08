import numpy as np


class GradientCalculator:
    """
    Calculate the gradients using the sobel operator
    """

    def __init__(self, array):
        self.x_grad = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.y_grad = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.array = array

    def _pad_with(self, vector, pad_width, iaxis, kwargs):
        """
        Helper function to pad arrays
        """
        pad = kwargs.get("padder", 0)
        vector[: pad_width[0]] = pad
        vector[-pad_width[1] :] = pad

    def _get_padded_array(self, array, kernel):
        """
        Return a padded array for convenience for convolution
        """
        return np.pad(array, len(kernel) // 2, self._pad_with, padder=0)

    def get_convolution(self, kernel):
        """
        Apply the convolution about the 2D Gaussian kernel of size self.kernel_size about array
        """
        padded = self._get_padded_array(self.array, kernel)
        r, c = self.array.shape
        convoluted = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                for k in range(len(kernel)):
                    convoluted[i][j] += np.dot(
                        kernel[k], padded[i + k][j : j + len(kernel)]
                    )
        return convoluted

    def _get_x_grad(self):
        return self.get_convolution(self.x_grad)

    def _get_y_grad(self):
        return self.get_convolution(self.y_grad)

    def get_magnitudes_and_angles(self):
        grad_x = self._get_x_grad()
        grad_y = self._get_y_grad()
        r = len(grad_x)
        c = len(grad_x[0])
        assert r == len(grad_y) and c == len(grad_y[0])  # Ensure validity
        magnitudes = np.sqrt(np.square(grad_x) + np.square(grad_y))
        angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
        return magnitudes, angles
