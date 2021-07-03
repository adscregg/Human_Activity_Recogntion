from torchvision.models import resnet101
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

from calculate_scattering import preprocessScatteringCoeffs
from dataset_classes import scatteringDataset, imageDataset
from models import ScatteringModel, MultiScalePretrained
from run import runModel
from identity import Identity

# ============================ SETUP ===============================================================

BATCH_SIZE = 64
DATA_DIR = './data/NTU_RGB+D/transformed_images'
WEIGHTS_PATH = './weights/'
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
loss_fn = CrossEntropyLoss()

def input_size(J, L):
    n = 1 + L*J + L**2 * J * (J - 1)/2
    n *= 3
    return int(n)



# create the image datasets with varying test sample sizes
images_train_5_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:5])
images_train_10_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:10])
images_train_15_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:15])
images_train_all_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS)
images_test = imageDataset(DATA_DIR, TEST_SUBJECT_IDS)

# create Dataloader generator objects for the images datasets
images_trainloader_5_subs = DataLoader(images_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_10_subs = DataLoader(images_train_10_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_15_subs = DataLoader(images_train_15_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_all_subs = DataLoader(images_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
images_testloader_all_subs = DataLoader(images_test, batch_size = BATCH_SIZE, shuffle = True)



scattering_dict_J4_L8 = preprocessScatteringCoeffs(DATA_DIR, J = 4, L = 8, batch_size = 64) # precalculate the flattened and pooled scattering coeffs
in_size_J4_L8 = input_size(4, 8)

# create the scattering datasets with varying test sample sizes
scattering_train_5_subs = scatteringDataset(scattering_dict_J4_L8, TRAIN_SUBJECT_IDS[:5])
scattering_train_10_subs = scatteringDataset(scattering_dict_J4_L8, TRAIN_SUBJECT_IDS[:10])
scattering_train_15_subs = scatteringDataset(scattering_dict_J4_L8, TRAIN_SUBJECT_IDS[:15])
scattering_train_all_subs = scatteringDataset(scattering_dict_J4_L8, TRAIN_SUBJECT_IDS)
scattering_test = scatteringDataset(scattering_dict_J4_L8, TEST_SUBJECT_IDS)

# create Dataloader generator objects for the scattering datasets
scattering_trainloader_5_subs = DataLoader(scattering_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_10_subs = DataLoader(scattering_train_10_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_15_subs = DataLoader(scattering_train_15_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_all_subs = DataLoader(scattering_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader_all_subs = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)





# ============================ TRAINING ===============================================================

ResNet = resnet101(pretrained = True)
ResNet.fc = Identity()
resnet_multiscale = MultiScalePretrained(ResNet).to(DEVICE)
weights_file = 'example.pth'
optimiser = optim.Adam(resnet_multiscale.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
resnet_multiscale_class = runModel(resnet_multiscale, DEVICE, optimiser, loss_fn, images_trainloader_all_subs, images_testloader_all_subs, scheduler)
# resnet_multiscale_class.train(epochs = 50, validate = True)
# resnet_multiscale_class.save_model(WEIGHTS_PATH+weights_file)
# print(resnet_multiscale_class.test_accuracy)



ScatNet = ScatteringModel(in_size_J4_L8, [256, 512, 512, 256]).to(DEVICE)
weights_file = 'example.pth'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader_all_subs, scheduler)
ScatNet_class.train(epochs = 500, validate = False)
# ScatNet_class.save_model(WEIGHTS_PATH+weights_file)
