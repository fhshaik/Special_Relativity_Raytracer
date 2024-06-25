import numpy as np
import math 
from abc import ABC, abstractmethod
from PIL import Image



class FourVector:
    def __init__(array):
        
        return


class Camera:
    maxdepth = 10
    def __init__(self):
        self.image_height = 256
        self.image_width = 256

        focal_length = 1.0
        viewport_width = 2.0
        viewport_height = 2.0

        viewport_u = np.array([viewport_width,0,0])
        viewport_v = np.array([0,-viewport_height,0])

        self.pixel_delta_u = viewport_u/self.image_width
        self.pixel_delta_v = viewport_v/self.image_width

        viewport_upper_left = -np.array([0,0,focal_length])-viewport_u/2 - viewport_v/2
        pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
        pixel00_locations = np.tile(pixel00_loc, (self.image_height, self.image_width, 1))
        indices_i = np.repeat(np.tile(np.arange(self.image_width), (self.image_height, 1))[:, :, np.newaxis], 3, axis=2)
        indices_j = np.repeat(np.tile(np.arange(self.image_height)[:, np.newaxis],  (1,self.image_width))[:, :, np.newaxis], 3, axis=2)
        #finally setting up our camera location through this
        self.pixel_centers = pixel00_locations+indices_i*self.pixel_delta_u+ indices_j*self.pixel_delta_v