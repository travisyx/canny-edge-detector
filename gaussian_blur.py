import numpy as np
from picture import Picture


class GaussianBlur:
    def __init__(self, sigma, kernel_size):
        """
        Notes:
            - Large sigma values are better for more noise
            - Kernel size values are normally odd, larger values (>= 5) for stronger blur
            - OpenCV sets the kernel size as int(3*sigma)
        """
        self.sigma = sigma
        self.kernel_size = kernel_size

    def _get_kernel1d(self):
        values = np.linspace(
            -self.kernel_size // 2, self.kernel_size // 2, self.kernel_size
        )
        kernel = np.exp(-(values**2) / (2 * self.sigma**2))
        return kernel / np.sum(kernel)

    def apply(self, image):
        """
        Returns: Gaussian blurred image of intensities
        """
        kernel = self._get_kernel1d()
        row = kernel.reshape(-1, 1)
        col = kernel
        blurred_row = np.apply_along_axis(
            lambda x: np.convolve(x, row.flatten(), mode="same"),
            1,
            image.get_intensities(),
        )
        blurred = np.apply_along_axis(
            lambda x: np.convolve(x, col, mode="same"), 0, blurred_row
        )
        return blurred
