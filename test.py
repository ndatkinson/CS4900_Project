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

transforms = transforms.Compose([transforms.ToPILImage(), 
		transforms.Resize((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
		transforms.ToTensor()])

def main():
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
	validationpath = open(config.IMAGE_NAMES_PATH+"//val.txt", "r")
	
	names = validationpath.readlines()
	validationpath.close()

	model = torch.load("unit_tgs_VOC2012.pth")
	model.eval()
	
	test_Image_Names = [str.replace(word, "\n", ".jpg") for word in names]
	mask_Image_Names = [str.replace(word, "\n", ".png") for word in names]
	
	testImages = [config.IMAGE_DATASET_PATH + word for word in test_Image_Names]
	testMasks = [config.MASK_DATASET_PATH + word for word in mask_Image_Names]

	testDS = SegmentationDataset(imagePaths=testImages,	maskPaths=testMasks,
				transforms=transforms)

	print(f"[INFO] found {len(testDS)} examples in the test set...")

	testLoader = DataLoader(testDS, shuffle=False,
            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
   
	unet = UNet().to(config.DEVICE)
	lossFunc = BCEWithLogitsLoss()
	opt = Adam(unet.parameters(),lr=config.INIT_LR)
 
	testSteps = len(testDS)
	H = {"train_loss": [], "test_loss":[]}

	print("[INFO] training the network...")
	startTime = time.time()
    
    
	#testing loop
	totalTestLoss = 0
	testSteps = len(testDS)
	
	with torch.no_grad():
		unet.eval()
		for(x,y) in testLoader:
			(x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			prediction = predict.make_prediction(model, x)
			predictionByModel = model(x)

			totalTestLoss += lossFunc(prediction, y)
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

    #torch.save(unet, config.MODEL_PATH)


if __name__ == "__main__":
    main()
