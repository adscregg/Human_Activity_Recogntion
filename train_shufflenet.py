from torchvision.models import resnet101, shufflenet_v2_x1_0
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

from calculate_scattering import preprocessScatteringCoeffs
from dataset_classes import imageDataset
from models import MultiScalePretrained
from run import runModel
from identity import Identity

print('Finished importing')

# ============================ SETUP ===============================================================

BATCH_SIZE = 64
DATA_DIR = './data/NTU_RGB+D/transformed_images'
WEIGHTS_PATH = './weights/'
SUMMARIES_PATH = './model_summaries/'
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
loss_fn = CrossEntropyLoss()



print('Creating image datasets...')
# create the image datasets with varying test sample sizes
images_train_5_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:5])
images_train_10_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:10])
images_train_15_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:15])
images_train_all_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS)
images_test = imageDataset(DATA_DIR, TEST_SUBJECT_IDS)
print('Done')

print('Creating image DataLoaders...')
# create Dataloader generator objects for the images datasets
images_trainloader_5_subs = DataLoader(images_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_10_subs = DataLoader(images_train_10_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_15_subs = DataLoader(images_train_15_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_all_subs = DataLoader(images_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
images_testloader = DataLoader(images_test, batch_size = BATCH_SIZE, shuffle = True)
print('Done')




# ============================ TRAINING ===============================================================


print('Training ShuffleNets:')
# ============== ShuffleNet =========================

# === 5 Subjects ===

ShuffleNet = shufflenet_v2_x1_0(pretrained = True)
# ShuffleNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ShuffleNet).to(DEVICE)
# weights_file = 'shufflenet_5_subs.pth'
# summary_file = 'shufflenet_5_subs.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_5_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True)
# MS_CNN_class.create_model_summary('ShuffleNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 10 Subjects ===

ShuffleNet = shufflenet_v2_x1_0(pretrained = True)
# ShuffleNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ShuffleNet).to(DEVICE)
# weights_file = 'shufflenet_10_subs.pth'
# summary_file = 'shufflenet_10_subs.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_10_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True)
# MS_CNN_class.create_model_summary('ShuffleNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 15 Subjects ===

ShuffleNet = shufflenet_v2_x1_0(pretrained = True)
# ShuffleNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ShuffleNet).to(DEVICE)
# weights_file = 'shufflenet_15_subs.pth'
# summary_file = 'shufflenet_15_subs.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_15_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True)
# MS_CNN_class.create_model_summary('ShuffleNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ShuffleNet = shufflenet_v2_x1_0(pretrained = True)
ShuffleNet.fc = Identity()
MS_CNN = MultiScalePretrained(ShuffleNet).to(DEVICE)
weights_file = 'shufflenet_all_subs.pth'
summary_file = 'shufflenet_all_subs.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_all_subs, images_testloader, scheduler)
MS_CNN_class.train(epochs = 50, validate = False)
MS_CNN_class.create_model_summary('ShuffleNet')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)
