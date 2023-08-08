import numpy as np
from convolutor import Convolutor
from picture import Picture

class GaussianBlur:
    def __init__(self, sigma, kernel_size):
        '''
        Notes:
            - Large sigma values are better for more noise
            - Kernel size values are normally odd, larger values (>= 5) for stronger blur
            - OpenCV sets the kernel size as int(3*sigma)
        '''
        self.sigma = sigma
        self.kernel_size = kernel_size

    def _calculate_gaussian(self, x, y):
        scale = 1/(2*np.pi*self.sigma**2)
        exponent = np.exp(-(x**2+y**2)/(2*self.sigma**2))
        return scale*exponent

    def _get_kernel(self):
        '''
        Returns: 2D Gaussian kernel of size self.kernel_size
        '''
        sz = self.kernel_size
        center = sz//2
        kernel = np.zeros((sz,sz))
        for i in range(-center, center+1):
            for j in range(-center, center+1):
                kernel[i+center][j+center] = self._calculate_gaussian(i,j)
        kernel /= np.sum(kernel)
        return kernel

    def apply(self, image):
        '''
        Returns: Gaussian blurred image of intensities
        '''
        arr = image.get_intensities()
        conv = Convolutor(arr,self._get_kernel())
        return conv.get_convolution()
