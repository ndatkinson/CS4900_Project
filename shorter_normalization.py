import numpy as np


#plug in data from Standard Deviation script 
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#test array - replace with different values when needed
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
