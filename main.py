from torchvision.models import resnet101
from torch.utils.data import DataLoader

from calculate_scattering import preprocessScatteringCoeffs
from dataset_classes import scatteringDataset, imageDataset
from models import ScatteringModel, MultiScalePretrained
from run import runModel
from identity import Identity

BATCH_SIZE = 64
WEIGHTS_PATH = './weights/'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
