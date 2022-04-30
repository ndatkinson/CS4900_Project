from torch.utils.data import Dataset
import cv2
from PIL import Image
#sets the paths of images, masks, and transforms
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
    #returns number of images in the image paths list
    def __len__(self):
        return len(self.imagePaths)
#defines paths for images and masks
    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath, 0)
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        mask = cv2.imread(self.maskPaths[idx], 0)
        mask = Image.fromarray(np.uint8(mask)).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)
