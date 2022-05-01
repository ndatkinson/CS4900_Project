from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import datasets
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import cv2
# adapted from https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#transforms = transforms.Compose([transforms.ToPILImage(),
#        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
#            config.INPUT_IMAGE_WIDTH)),
#        transforms.ToTensor()])
#transform for converting image to a tensor.
transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.3952, 0.4302, 0.4467], [0.2837, 0.2719, 0.2770])])
#color dictionary for color transforms
Color_Dict = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128), 
              5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (64, 0, 0), 9: (192, 0, 0), 
              10: (64, 128, 0), 11: (192, 128, 0), 12: (64, 0, 128), 13: (192, 0, 128), 14: (64, 128, 128), 
              15: (192, 128, 128), 16: (0, 64, 0), 17: (128, 64, 0), 18: (0, 192, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

def main():
     #load image file paths into script 
    imageNamesFile = open(config.IMAGE_NAMES_PATH + "//train.txt", "r")
    names = imageNamesFile.readlines()
    imageNamesFile.close()
        #makes paths for images to get complete path of image
    train_images_names = [str.replace(word, '\n', '.jpg') for word in names]
    mask_images_names = [str.replace(word, '\n', '.png') for word in names]
    #loads images from paths. One for images and the other for masks
    trainImages = [config.IMAGE_DATASET_PATH + word for word in train_images_names]
    trainMasks = [config.MASK_DATASET_PATH + word for word in mask_images_names]

    #split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

    #(trainImages, testImages) = split[:2]
    #(trainMasks, testMasks) = split[2:]
        
    #print("[INFO] saving testing image paths...")
    #f = open(config.TEST_PATHS, "w")
    #f.write("\n".join(testImages))
    #f.close()
    #initializes trainDS with paths for images and paths for masks
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                transforms=transforms)
    
    #prints how many files are in the trainDS
    #testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,transforms=transforms)
    #testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set>>>")
    #print(f"[INFO] found {len(testDS)} exanmples in the test set...")
     #initializes trainloader with train DS, batch size, and memory
    trainLoader = DataLoader(trainDS, shuffle=True,
            batch_size = config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            num_workers = os.cpu_count())
    #testLoader = DataLoader(testDS, shuffle=False,
    #        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    #        num_workers=os.cpu_count())
     #instantialing model and setting its workers to the most accessible device
    unet = UNet().to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(),lr=config.INIT_LR)
    trainSteps = len(trainDS)
    #testSteps = len(testDS)
    H = {"train_loss": [], "test_loss":[]}
     #training loop
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0

        #loops through trainloader
        for(i, (x, y)) in enumerate(trainLoader):
            (x,Y)=(x.to(config.DEVICE), y.to(config.DEVICE))

            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            #record train loss
            totalTrainLoss +=loss

            with torch.no_grad():
                unet.eval()

 #               for(x,y) in testLoader:
 #                   (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
#
 #                   pred = unet(x)
  #                  totalTestLoss+=lossFunc(pred,y)
               #train loss is assessed and recorded.
            avgTrainLoss = totalTrainLoss /trainSteps
  #          avgTestLoss = totalTestLoss/testSteps
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
  #          H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

            print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
            print("Train loss: {:6f}".format(
                avgTrainLoss))

    endTime = time.time()
    print("[INFO] total time taken to trian the model: {:.2f}s".format(endTime-startTime))

       #training loss is graphed
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc ="lower left")
    plt.savefig(config.PLOT_PATH)
      #model is saved into the path specified in the config file
    torch.save(unet, config.MODEL_PATH)

if __name__ == "__main__":
    main()
