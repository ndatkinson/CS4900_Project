from matplotlib.transforms import Transform
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
import numpy as np

transform = transforms.Compose([
		#transforms.CenterCrop((config.INPUT_IMAGE_HEIGHT,
		#	config.INPUT_IMAGE_WIDTH)),
		transforms.Resize((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.456], std = [0.224])
		#transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
		])
		
DSTransform = transforms.Compose([transforms.ToTensor()])

transformToImage = transforms.Compose([
	transforms.ToPILImage(), 
		transforms.CenterCrop((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH))])
def main():
	transformToImage = transforms.Compose([
		transforms.CenterCrop((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
			transforms.ToPILImage()])
	
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
	validationpath = open(config.IMAGE_NAMES_PATH+"//val.txt", "r")

	names = validationpath.readlines()
	validationpath.close()
	model = torch.load("unit_tgs_VOC2012.pt")
	model.eval()
	
	validation_Image_Names = [str.replace(word, "\n", ".jpg") for word in names]
	mask_Image_Names = [str.replace(word, "\n", ".png") for word in names]
	
	validationImages = [config.IMAGE_DATASET_PATH + word for word in validation_Image_Names]
	validationMasks = [config.MASK_DATASET_PATH + word for word in mask_Image_Names]
	
	validationDS = SegmentationDataset(imagePaths=validationImages,	maskPaths=validationMasks,
				transforms=transform)
				
	validationLoader = DataLoader(validationDS, shuffle=False,
				batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			num_workers=0)
        
	totalValidationLoss = 0
	validationSteps = len(validationDS)
	
	unet = UNet().to(config.DEVICE)
	lossFunc = BCEWithLogitsLoss()
	opt = Adam(unet.parameters(),lr=config.INIT_LR)
	
	with torch.no_grad():
		model.eval()
		for x,y in validationLoader:
			(x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			print(x.size())
			
			
			print(type(x))

			H = {"train_loss": [], "validation_loss":[]}
			prediction = model(x)
			#image_index = testImages.index()
		

			totalValidationLoss += lossFunc(prediction, y)
			avgValidationLoss = totalValidationLoss/validationSteps
			H["validation_loss"].append(avgValidationLoss.cpu().detach().numpy())
			print("Valdiation Loss : {:4f}".format(
				avgValidationLoss))


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