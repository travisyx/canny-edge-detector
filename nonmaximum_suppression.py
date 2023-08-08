import numpy as np


class NonMaximumSuppression:
    def __init__(self, magnitudes, angles):
        self.magnitudes = magnitudes
        self.angles = (angles + 180) % 180  # Convert range from [-180,180] to [0,180]

    def binning(self):
        """
        Returns an array of values in the range [0,3] responding to the direction of the edges
        These values correspond to:
            0: horizontal edges
            1: positive diagonal edges (45 degrees)
            2: vertical edges
            3: negative diagonal edges (other diagonal)
        """
        binned_arr = np.zeros_like(self.angles)
        bins = [0, 45, 90, 135, 180]
        for i in range(4):
            binned_arr[
                np.logical_and(self.angles >= bins[i], self.angles < bins[i + 1])
            ] = i
        return binned_arr

    def apply_suppression(self):
        """
        Apply the nonmaximum suppression: suppress gradient values that are not the local maxima (edges)
        For the border, it is common to set border values to 0
        """
        suppressed = np.copy(self.magnitudes)
        bins = self.binning()
        for r in range(1, len(self.magnitudes) - 1):
            for c in range(1, len(self.magnitudes[0]) - 1):
                bin_val = bins[r][c]
                if bin_val == 0:  # Horizontal edge
                    if (
                        self.magnitudes[r][c] < self.magnitudes[r][c - 1]
                        or self.magnitudes[r][c] < self.magnitudes[r][c + 1]
                    ):
                        suppressed[r][c] = 0
                elif bin_val == 1:  # Positive diagonal
                    if (
                        self.magnitudes[r][c] < self.magnitudes[r - 1][c - 1]
                        or self.magnitudes[r][c] < self.magnitudes[r + 1][c + 1]
                    ):
                        suppressed[r][c] = 0
                elif bin_val == 2:  # Vertical edge
                    if (
                        self.magnitudes[r][c] < self.magnitudes[r - 1][c]
                        or self.magnitudes[r][c] < self.magnitudes[r + 1][c]
                    ):
                        suppressed[r][c] = 0
                else:  # Negative diagonal
                    if (
                        self.magnitudes[r][c] < self.magnitudes[r - 1][c + 1]
                        or self.magnitudes[r][c] < self.magnitudes[r + 1][c - 1]
                    ):
                        suppressed[r][c] = 0

        # Set borders to 0
        suppressed[0, :] = 0
        suppressed[-1, :] = 0
        suppressed[:, 0] = 0
        suppressed[:, -1] = 0
        return suppressed
