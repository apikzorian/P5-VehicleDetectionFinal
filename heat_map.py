import numpy as np

from scipy.ndimage.measurements import label

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

# Class that process heatmap and stores bounding boxes from heatmap of previous frame
class HeatMap():

    def __init__(self, img):
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        self.cold = 1
        self.hot = 1
        self.saturation = 10
        self.boundingBoxes = []
        self.threshold = 3

    # Heats every pixel in bounding box by one increment of self.hot. Begins by chilling all pixels
    #
    def add_heat(self, bbox_list):
        self.chill()
        if len(self.boundingBoxes) == 0:
            for box in bbox_list:
                self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.hot
                self.heatmap[np.where(self.heatmap > self.saturation)] = self.saturation
        else:
            for box in bbox_list:
                heat = 0
                self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.hot
                for box_exist in self.boundingBoxes:
                    if box == box_exist:
                        self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.hot
                        heat = 1
                        break

        self.heatmap[np.where(self.heatmap > self.saturation)] = self.saturation
        self.boundingBoxes = bbox_list
        return self.heatmap

    # Reduce all pixels in heatmap by 1 factor of self.cold and revert anything below 0 to 0
    def chill(self):
        self.heatmap -= self.cold
        self.heatmap[np.where(self.heatmap < 0)] = 0

    # Returns self.heatmap
    def get_heat(self):
        return np.copy(self.heatmap)

    # Zero out pixels below the threshold and returns threshold map
    def apply_threshold(self):
        self.heatmap[self.heatmap <= self.threshold] = 0
        return self.heatmap

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_boxes(self):
        return self.boundingBoxes
