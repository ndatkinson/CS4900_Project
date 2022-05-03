from ctypes.wintypes import RGB
from matplotlib.transforms import Transform
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.nn import Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
from PIL import Image
import predict
import numpy as np

from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
import cv2





model = torch.load("C:\\Users\\noahd\\Desktop\\Senior Seminar\\Unet_10_Epochs")

im  = Image.open("C:\\Users\\noahd\\Desktop\\Senior Seminar\\training\\Dataset\\VOC2012\\JPEGImages\\2007_000033.jpg")


preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.3952, 0.4302, 0.4467], std = [0.2837, 0.2719, 0.2770])
])

input_tensor = preprocess(im)
input_batch = input_tensor.unsqueeze(0)

model.eval()
output = model(input_batch)[0]


postprocess = transforms.Compose([
    transforms.Normalize(mean = [0.3952, 0.4302, 0.4467], std = [0.2837, 0.2719, 0.2770]),
    transforms.ToPILImage()])

softmax = Softmax(dim=1)
final_output = postprocess(softmax(output))
final_output.show()