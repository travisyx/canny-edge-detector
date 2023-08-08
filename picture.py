import numpy as np
import sys
from PIL import Image

class Picture():
    def __init__(self, url, x_dim=500, y_dim=500):
        '''
        Notes:
            - Large sigma values are better for more noise
            - Kernel size values are normally odd, larger values (>= 5) for stronger blur
            - OpenCV sets the kernel size as int(3*sigma), which is what is used here
            - PIL outputs image in RGB format
        '''
        self.url = url
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.have_intensities = False
        try:
            self.img = np.array(Image.open(url).resize((x_dim,y_dim)))
        except:
            print("Please make sure you are using a valid image")
            sys.exit(1)
            
        self.intensities = np.zeros((x_dim,y_dim))
        
    def _grayscale_simple(self):
        '''
        Fills in the intensity array with the average of each RGB value
        
        Returns: none
        '''
        if self.have_intensities:
            return
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                self.intensities[i][j] = np.full((1,3),np.mean(self.img[i][j]))
        self.have_intensities = True
    
    def _grayscale_luma(self):
        '''
        Fills in the intensity array using the Luma formula (based on the ITU-R BT.709 recommendation) 
        to correct for the human eye
        See: https://en.wikipedia.org/wiki/Rec._709

        Returns: none
        '''
        if self.have_intensities:
            return
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                tmp = self.img[i][j]
                self.intensities[i][j] = tmp[0]*0.2126+tmp[1]*0.7152+tmp[2]*0.0722
        self.have_intensities = True
    
    def _image_from_intensities(self):
        '''
        Returns: a numpy array of dimensions (self.x_dim, self.y_dim, 3) with the R, G, and B values at each pixel 
        being that of the intensity
        '''
        arr = np.zeros((self.x_dim,self.y_dim,3))
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                val = self.intensities[i][j]
                arr[i][j] = np.array([val,val,val])
        return arr
    
    def get_image(self):
        return self.img
    
    def get_intensities(self):
        if not self.have_intensities:
            self._grayscale_luma()
            self.have_intensities = True
        return self.intensities
    
    def save_image(self, url, data):
        im = Image.fromarray(data.astype(np.uint8))
        im.save(url)
    
    def show_image(self):
        image = Image.fromarray(self.img)
        image.show()
    
    def show_grayscale(self):
        if not self.have_intensities:
            self._grayscale_luma()
        image = Image.fromarray(self._image_from_intensities().astype(np.uint8))
        image.show()

    def show_image_from_intensities(self, intensities):
        image = Image.fromarray(intensities.astype(np.uint8))
        image.show()