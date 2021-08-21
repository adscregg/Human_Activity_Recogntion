from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

# from calculate_scattering import preprocessScatteringCoeffs
from dataset_classes import scatteringDataset
from models import ScatteringModel
from run import runModel


# ============================ SETUP ===============================================================

BATCH_SIZE = 128
WEIGHTS_PATH = './weights/NTU-RGB+D/J2_L8_44/ScatNet_Deep/'
SUMMARIES_PATH = './model_summaries/NTU-RGB+D/J2_L8_44/ScatNet_Deep/'
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
loss_fn = CrossEntropyLoss()

def input_size(J, L):
    n = 1 + L*J + L**2 * J * (J - 1)/2
    n *= 3
    return int(n)


scattering_dir = 'C:/Local/scattering coeffs_J2_single_scale/' # note, the J2 files are not single scale
in_size_J4_L8 = input_size(4, 8)
in_size_J2_L8_44 = input_size(2,8) * 4*4
in_size_J5_L8 = input_size(5, 8)

# create the scattering datasets with varying test sample sizes
scattering_train_2_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:2])
scattering_train_5_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:5])
scattering_train_12_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:12])
scattering_train_all_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS)
scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS)

# create Dataloader generator objects for the scattering datasets
scattering_trainloader_2_subs = DataLoader(scattering_train_2_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_5_subs = DataLoader(scattering_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_12_subs = DataLoader(scattering_train_12_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_all_subs = DataLoader(scattering_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)




# ============================ TRAINING ===============================================================


print('Training Scattering Models:')
# ============  Scattering =====================

layers = [512, 1024, 1024, 512, 256, 'D']

# === 2 Subjects ===

ScatNet = ScatteringModel(in_size_J2_L8_44, layers).to(DEVICE)
weights_file = 'scattering_2_subs.pth'
summary_file = 'scattering_2_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_2_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Scattering (2)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)




# === 5 Subjects ===

ScatNet = ScatteringModel(in_size_J2_L8_44, layers).to(DEVICE)
weights_file = 'scattering_5_subs.pth'
summary_file = 'scattering_5_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_5_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Scattering (5)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === 12 Subjects ===

ScatNet = ScatteringModel(in_size_J2_L8_44, layers).to(DEVICE)
weights_file = 'scattering_12_subs.pth'
summary_file = 'scattering_12_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_12_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Scattering (12)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# === All Subjects ===

ScatNet = ScatteringModel(in_size_J2_L8_44, layers).to(DEVICE)
weights_file = 'scattering_all_subs.pth'
summary_file = 'scattering_all_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Scattering (all)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
