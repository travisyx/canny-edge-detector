import numpy as np


class Hysteresis:
    """
    Apply edge tracking via hysteresis
    """

    def __init__(self, intensities, mapping):
        self.intensities = intensities
        self.mapping = mapping

    def _hysteresis(self):
        """
        Promote weak edges (0.5) to strong if it is adjacent to a strong edge in the 8 directions
        """
        row, col = self.mapping.shape
        dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for r in range(1, row - 1):
            for c in range(1, col - 1):
                if self.mapping[r][c] == 0.5:
                    for d in dirs:
                        if self.mapping[r + d[0]][c + d[1]] == 1:
                            self.mapping[r][c] = 1

    def apply(self):
        """
        Apply edge tracking by hysteresis

        Returns the intensities of strong edges
        """
        self._hysteresis()
        mapping = self.mapping == 1
        row, col = self.intensities.shape
        strong_edges = np.copy(self.intensities)
        for r in range(row):
            for c in range(col):
                if mapping[r][c] != 1:
                    strong_edges[r][c] = 0
        return strong_edges
