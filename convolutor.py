import numpy as np


class Convolutor:
    """
    Helper class to perform the convolution array*kernel
    """

    def __init__(self, array, kernel):
        assert len(kernel) == len(kernel[0])  # Ensure the kernel is valid
        self.array = array
        self.kernel = kernel

    def _pad_with(self, vector, pad_width, iaxis, kwargs):
        """
        Helper function to pad arrays
        """
        pad = kwargs.get("padder", 0)
        vector[: pad_width[0]] = pad
        vector[-pad_width[1] :] = pad

    def _get_padded_array(self, array):
        """
        Return a padded array for convenience for convolution
        """
        return np.pad(array, len(self.kernel) // 2, self._pad_with, padder=0)

    def get_convolution(self):
        """
        Apply the convolution about the 2D Gaussian kernel of size self.kernel_size about array
        """
        # TODO: Use FFT
        padded = self._get_padded_array(self.array)
        r = len(self.array)
        c = len(self.array[0])
        convoluted = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                for k in range(len(self.kernel)):
                    convoluted[i][j] += np.dot(
                        self.kernel[k], padded[i + k][j : j + len(self.kernel)]
                    )
        return convoluted
