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

Color_Dict = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128), 
              5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (64, 0, 0), 9: (192, 0, 0), 
              10: (64, 128, 0), 11: (192, 128, 0), 12: (64, 0, 128), 13: (192, 0, 128), 14: (64, 128, 128), 
              15: (192, 128, 128), 16: (0, 64, 0), 17: (128, 64, 0), 18: (0, 192, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

def main():
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set>>>")
    print(f"[INFO] found {len(testDS)} exanmples in the test set...")

    trainLoader = DataLoader(trainDS, shuffle=True,
            batch_size = config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            num_workers = os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
            num_workers=os.cpu_count())

    unet = UNet().to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(),lr=config.INIT_LR)
    trainSteps = len(trainDS)
    testSteps = len(testDS)
    H = {"train_loss": [], "test_loss":[]}

    print("[INFO] training the network...")
    startTime = time.time()
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

            avgTrainLoss = totalTrainLoss /trainSteps
            avgTestLoss = totalTestLoss/testSteps
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

            print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
            print("Train loss: {:6f}, Test loss: {:.4f}".format(
                avgTrainLoss, avgTestLoss))

    endTime - time.time()
    print("[INFO] total time taken to trian the model: {:.2f}s".format(endtime-startTime))


    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylael("Loss")
    plt.legend(loc ="lower left")
    plt.savefig(config.PLOT_PATH)

    torch.save(unet, config.MODEL_PATH)

if __name__ == "__main__":
    main()
