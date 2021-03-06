from torchvision.models import resnet101
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim

from dataset_classes import imageDataset
from models import MultiScalePretrained
from run import runModel
from identity import Identity


# ============================ SETUP ===============================================================

BATCH_SIZE = 128
DATA_DIR = 'C:/Local/transformed_images/'
WEIGHTS_PATH = './weights/ResNet/'
SUMMARIES_PATH = './model_summaries/ResNet/'
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
loss_fn = CrossEntropyLoss()




# create the image datasets with varying test sample sizes
images_train_2_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:2])
images_train_5_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:5])
images_train_12_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS[:12])
images_train_all_subs = imageDataset(DATA_DIR, TRAIN_SUBJECT_IDS)
images_test = imageDataset(DATA_DIR, TEST_SUBJECT_IDS)

# create Dataloader generator objects for the images datasets
images_trainloader_2_subs = DataLoader(images_train_2_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_5_subs = DataLoader(images_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_12_subs = DataLoader(images_train_12_subs, batch_size = BATCH_SIZE, shuffle = True)
images_trainloader_all_subs = DataLoader(images_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
images_testloader = DataLoader(images_test, batch_size = BATCH_SIZE, shuffle = True)

# ============================ TRAINING ===============================================================

print('Training ResNets:')
# ============== ResNet =========================

# === 2 Subjects ===

ResNet = resnet101(pretrained = True)
ResNet.fc = Identity()
MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
weights_file = 'resnet_2_subs.pth'
summary_file = 'resnet_2_subs.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_2_subs, images_testloader, scheduler)
MS_CNN_class.train(epochs = 100, validate = True) # start training loop
MS_CNN_class.create_model_summary('ResNet (2)')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 5 Subjects ===

ResNet = resnet101(pretrained = True) # define model architecture
ResNet.fc = Identity() # remove final classification layer so multi-layer stucture can be implemented
MS_CNN = MultiScalePretrained(ResNet).to(DEVICE) # create multi-layer architecture model
weights_file = 'resnet_5_subs.pth' # file name of the weights of the model
summary_file = 'resnet_5_subs.json' # file name of the model summary
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001) # define the optimiser
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_5_subs, images_testloader, scheduler)
MS_CNN_class.train(epochs = 100, validate = True) # start training loop
MS_CNN_class.create_model_summary('ResNet (5)') # create the model summary
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file) # save model weights
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file) # save model summary


# === 12 Subjects ===

ResNet = resnet101(pretrained = True)
ResNet.fc = Identity()
MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
weights_file = 'resnet_12_subs.pth'
summary_file = 'resnet_12_subs.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_12_subs, images_testloader, scheduler)
MS_CNN_class.train(epochs = 100, validate = True) # start training loop
MS_CNN_class.create_model_summary('ResNet (12)')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ResNet = resnet101(pretrained = True)
ResNet.fc = Identity()
MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
weights_file = 'resnet_all_subs.pth'
summary_file = 'resnet_all_subs.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_all_subs, images_testloader, scheduler)
MS_CNN_class.train(epochs = 100, validate = True) # start training loop
MS_CNN_class.create_model_summary('ResNet (all)')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)
