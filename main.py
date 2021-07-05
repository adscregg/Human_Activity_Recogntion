from torchvision.models import resnet101, squeezenet1_1, densenet121, shufflenet_v2_x1_0
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

from calculate_scattering import preprocessScatteringCoeffs
from dataset_classes import scatteringDataset, imageDataset
from models import ScatteringModel, MultiScalePretrained
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

def input_size(J, L):
    n = 1 + L*J + L**2 * J * (J - 1)/2
    n *= 3
    return int(n)



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


# scattering_dict_J4_L8 = preprocessScatteringCoeffs(DATA_DIR, save_dir = None, J = 4, L = 8, batch_size = 64) # precalculate the flattened and pooled scattering coeffs
scattering_dir = './data/NTU_RGB+D/scattering_coeffs/'
in_size_J4_L8 = input_size(4, 8)

print('Creating scattering datasets...')
# create the scattering datasets with varying test sample sizes
scattering_train_5_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:5])
scattering_train_10_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:10])
scattering_train_15_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:15])
scattering_train_all_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS)
scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS)
print('Done')

print('Creating scattering DataLoaders...')
# create Dataloader generator objects for the scattering datasets
scattering_trainloader_5_subs = DataLoader(scattering_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_10_subs = DataLoader(scattering_train_10_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_15_subs = DataLoader(scattering_train_15_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_all_subs = DataLoader(scattering_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)
print('Done')




# ============================ TRAINING ===============================================================

print('Training ResNets:')
# ============== ResNet =========================

# === 5 Subjects ===

ResNet = resnet101(pretrained = True) # define model architecture
# ResNet.fc = Identity() # remove final classification layer so multi-layer stucture can be implemented
# MS_CNN = MultiScalePretrained(ResNet).to(DEVICE) # create multi-layer architecture model
# weights_file = 'resnet_5_subs.pth' # file name of the weights of the model
# summary_file = 'resnet_5_subs.json' # file name of the model summary
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001) # define the optimiser
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser) # define the scheduler (optional, will change learning rate during training if supplied to runModel)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_5_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True) # start training loop
# MS_CNN_class.create_model_summary('ResNet') # create the model summary
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file) # save model weights
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file) # save model summary


# === 10 Subjects ===

ResNet = resnet101(pretrained = True)
# ResNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
# weights_file = 'resnet_10_subs.pth'
# summary_file = 'resnet_10_subs.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_10_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True)
# MS_CNN_class.create_model_summary('ResNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 15 Subjects ===

ResNet = resnet101(pretrained = True)
# ResNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
# weights_file = 'resnet_15_subs.pth'
# summary_file = 'resnet_15_subs.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_15_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = True)
# MS_CNN_class.create_model_summary('ResNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ResNet = resnet101(pretrained = True)
ResNet.fc = Identity()
MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
weights_file = 'resnet_all_subs.pth'
summary_file = 'resnet_all_subs.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader_all_subs, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 50, validate = False)
MS_CNN_class.create_model_summary('ResNet')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)


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
# MS_CNN_class.train(epochs = 50, validate = False)
MS_CNN_class.create_model_summary('ShuffleNet')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)



print('Training Linear Scattering Models:')
# =========== Linear Scattering ====================

# === 5 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8).to(DEVICE)
# weights_file = 'lin_scattering_5_subs.pth'
# summary_file = 'lin_scattering_5_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Lin Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 10 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8).to(DEVICE)
# weights_file = 'lin_scattering_10_subs.pth'
# summary_file = 'lin_scattering_10_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Lin Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 15 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8).to(DEVICE)
# weights_file = 'lin_scattering_15_subs.pth'
# summary_file = 'lin_scattering_15_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_15_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Lin Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8).to(DEVICE)
weights_file = 'lin_scattering_all_subs.pth'
summary_file = 'lin_scattering_all_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 50, validate = False)
ScatNet_class.create_model_summary('Lin Scattering')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


print('Training Shallow Scattering Models:')
# ============ Shallow Scattering =====================

layers = [256, 512]

# === 5 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'shallow_scattering_5_subs.pth'
# summary_file = 'shallow_scattering_5_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Shallow Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 10 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'shallow_scattering_10_subs.pth'
# summary_file = 'shallow_scattering_10_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Shallow Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 15 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'shallow_scattering_15_subs.pth'
# summary_file = 'shallow_scattering_15_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_15_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Shallow Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
weights_file = 'shallow_scattering_all_subs.pth'
summary_file = 'shallow_scattering_all_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 50, validate = False)
ScatNet_class.create_model_summary('Shallow Scattering')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


print('Training Deep Scattering Models:')
# ============ Deep Scattering =====================

layers = [256, 512, 512, 256]

# === 5 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'deep_scattering_5_subs.pth'
# summary_file = 'deep_scattering_5_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Deep Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 10 Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'deep_scattering_10_subs.pth'
# summary_file = 'deep_scattering_10_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_10_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Deep Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 15 Subjects ===

# ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
# weights_file = 'deep_scattering_15_subs.pth'
# summary_file = 'deep_scattering_15_subs.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_15_subs, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 50, validate = True)
# ScatNet_class.create_model_summary('Deep Scattering')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ScatNet = ScatteringModel(in_size_J4_L8, layers).to(DEVICE)
weights_file = 'deep_scattering_all_subs.pth'
summary_file = 'deep_scattering_all_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 50, validate = False)
ScatNet_class.create_model_summary('Deep Scattering')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
