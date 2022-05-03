from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
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
import numpy

def main():
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
	testingpath = open(config.IMAGE_NAMES_PATH+"//test.txt", "r")
	
	names = testingpath.readlines()
	testingpath.close()

	model = torch.load("unit_tgs_VOC2012.pth")
	model.eval()
	
	test_Image_Names = [str.replace(word, "\n", ".jpg") for word in names]
	mask_Image_Names = [str.replace(word, "\n", ".png") for word in names]
	
	testImages = [config.IMAGE_DATASET_PATH + word for word in test_Image_Names]
	testMasks = [config.MASK_DATASET_PATH + word for word in mask_Image_Names]
	
	startTime = time.time()
	with torch.no_grad():
		for i in range(len(test_Image_Names)):
			prediction = model(test_Image_Names[i])
		
		
			totalTestLoss += lossFunc(prediction, mask_Image_Names[i])
			avgTestLoss = totalTestLoss/testSteps
			H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
			print("Test Loss : {:4f}".format(
				avgTestLoss))
				
				
	endTime = time.time()
	print("[INFO] total time taken to test the model: {:.2f}s".format(endTime-startTime))

    #Post training graph of trainloss
	plt.style.use("ggplot")
	plt.figure()
    #plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["test_loss"], label="test_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc ="lower left")
	plt.savefig(config.PLOT_PATH)
	
if __name__ == "__main__":
	main()