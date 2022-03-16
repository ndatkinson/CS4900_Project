import PIL
from PIL import Image
import numpy as np


def color_map_info(palette):
    #list of classes
    labels = [
          'background', #0
          'aeroplane', #1
          'bicycle', #2
          'bird', #3
          'boat', #4
          'bottle', #5
          'bus', #6
          'car', #7
          'cat', #8
          'chair', #9
          'cow', #10
          'diningtable', #11
          'dog', #12
          'horse', #13
          'motorbike', #14
          'person', #15
          'pottedplant', #16
          'sheep', #17
          'sofa', #18
          'train', #19
          'tv/monitor', #20
          "void/unlabelled", #255
          ] 

    #Creating map of classes and corresponding RGB values
    Color_Values = {}
    for i in range(0,21*3,3):
        Color_Values[int(i/3)] = (palette[i], palette[i+1],palette[i+2])

    return(Color_Values)




def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

palette1 = color_map(N = 256)
palette1 = np.reshape(palette1, [-1,]) # reshape to 1-D array;
print(color_map_info(palette1))
