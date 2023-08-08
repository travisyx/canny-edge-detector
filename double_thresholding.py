import numpy as np

class DoubleThresholding:
    '''
    Apply double thresholding to classify edges as strong, weak, or non-edges and apply edge tracking by hysteresis
    '''
    def __init__(self, magnitudes, lower_threshold, upper_threshold):
        self.magnitudes = magnitudes
        self.lower = lower_threshold
        self.upper = upper_threshold

    def apply(self):
        '''
        Performs edge tracking by hysteresis
        '''
        strong = self.magnitudes >= self.upper
        weak = (self.magnitudes < self.upper) & (self.magnitudes >= self.lower)
        mapping = strong + weak*0.5
        return mapping
