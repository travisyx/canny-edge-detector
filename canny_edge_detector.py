import numpy as np
from double_thresholding import DoubleThresholding
from gaussian_blur import GaussianBlur
from gradient_calculator import GradientCalculator
from hysteresis import Hysteresis
from nonmaximum_suppression import NonMaximumSuppression
from picture import Picture


class CannyEdgeDetector:
    def __init__(
        self,
        url,
        sigma=0.84049642,
        kernel_size=9,
        lower_threshold=50,
        upper_threshold=100,
    ):
        self.picture = Picture(url)
        self.gaussian_blur = GaussianBlur(sigma, kernel_size)
        self.intensities = self.gaussian_blur.apply(self.picture)
        self.gradient_calculator = GradientCalculator(self.intensities)
        magnitudes, angles = self.gradient_calculator.get_magnitudes_and_angles()
        self.nonmaximum_suppression = NonMaximumSuppression(magnitudes, angles)
        thinned = self.nonmaximum_suppression.apply_suppression()
        self.double_thresholding = DoubleThresholding(
            thinned, lower_threshold, upper_threshold
        )
        mapping = self.double_thresholding.apply()
        self.hysteresis = Hysteresis(thinned, mapping)
        self.strong_edges = self.hysteresis.apply()

    def apply(self):
        self.picture.show_image_from_intensities(self.strong_edges)

    def save(self, url):
        self.picture.save_image(url, self.strong_edges)
