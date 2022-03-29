# -*- coding: utf-8 -*-
"""shorter normalization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cw6zyQxPYUW8CAb-yBqFIrcfe3FPQdKG
"""

import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

X = np.array([
    [ 0,  1],
    [ 2,  3],
    [ 4,  5],
    [ 6,  7],
    [ 8,  9],
    [10, 11],
    [12, 13],
    [14, 15]
])

scaled_x = NormalizeData(X)

print(scaled_x)