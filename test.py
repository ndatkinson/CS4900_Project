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

    split the data according to seed 9999 and a terst split that uses the first 12000 images of the dataset to train
    #split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=9999)

    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    
    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    #splits masks and pictures 
    #trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set>>>")
    print(f"[INFO] found {len(testDS)} exanmples in the test set...")
    #Sets batch size in train loader
    #trainLoader = DataLoader(trainDS, shuffle=True,
            #batch_size = config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            #num_workers = os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            num_workers=os.cpu_count())


   
    unet = UNet().to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(),lr=config.INIT_LR)
    #trainSteps = len(trainDS)
    testSteps = len(testDS)
    H = {"train_loss": [], "test_loss":[]}


    print("[INFO] training the network...")
    startTime = time.time()
    #training loop
    for e in tqdm(range(config.NUM_EPOCHS)):
        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0

        for(i, (x, y)) in enumerate(trainLoader):
            (x,Y)=(x.to(config.DEVICE), y.to(config.DEVICE))

            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss +=loss
            
            with torch.no_grad():
                unet.eval()

                for(x,y) in testLoader:
                    (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                    pred = unet(x)
                    totalTestLoss+=lossFunc(pred,y)

            #avgTrainLoss = totalTrainLoss /trainSteps
            avgTestLoss = totalTestLoss/testSteps
            #H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            
            
            print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
            print("Test loss: {:4f},".format(
               , avgTestLoss))

    endTime = time.time()
    print("[INFO] total time taken to trian the model: {:.2f}s".format(endTime-startTime))

    #Post training graph of trainloss
    print("Average train loss: " + avgTrainLoss);
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