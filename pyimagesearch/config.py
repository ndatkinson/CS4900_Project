import torch
import os

DATASET_PATH = os.path.join("Dataset", "VOC2012")

IMAGE_NAMES_PATH = os.path.join(DATASET_PATH, "ImageSets", "Segmentation")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "JPEGImages//")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "SegmentationClass//")

TEST_SPLIT = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.1
NUM_EPOCHS = 40
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH = 400
INPUT_IMAGE_HEIGHT = 300

THRESHOLD = 0.5

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unit_tgs_VOC2012.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_pths.txt"])

