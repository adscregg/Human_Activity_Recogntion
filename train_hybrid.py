from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim

from dataset_classes import scatteringDataset
from models import ScatNetCNNHybrid
from runHybrid import runModel


BATCH_SIZE = 128
WEIGHTS_PATH = './weights/Hybrid/'
SUMMARIES_PATH = './model_summaries/Hybrid/'
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_SUBJECT_IDS = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 38, 37, 39, 40]
loss_fn = CrossEntropyLoss()

scattering_dir = 'C:/Local/scattering_coeffs_not_pooled/'

# create the scattering datasets with varying test sample sizes
scattering_train_2_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:2], multi = False)
scattering_train_5_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:5], multi = False)
scattering_train_12_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS[:12], multi = False)
scattering_train_all_subs = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS, multi = False)
scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS, multi = False)

# create Dataloader generator objects for the scattering datasets
scattering_trainloader_2_subs = DataLoader(scattering_train_2_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_5_subs = DataLoader(scattering_train_5_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_12_subs = DataLoader(scattering_train_12_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_trainloader_all_subs = DataLoader(scattering_train_all_subs, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)


# ==== 2 Subjects =====

ScatNet = ScatNetCNNHybrid().to(DEVICE)
weights_file = 'hybrid_2_subs.pth'
summary_file = 'hybrid_2_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_2_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid (2)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)

# ==== 5 subs ====

ScatNet = ScatNetCNNHybrid().to(DEVICE)
weights_file = 'hybrid_5_subs.pth'
summary_file = 'hybrid_5_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_5_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid (5)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)

# ===== 12 subs ====

ScatNet = ScatNetCNNHybrid().to(DEVICE)
weights_file = 'hybrid_12_subs.pth'
summary_file = 'hybrid_12_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_12_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid (12)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)


# ==== 20 subs ====

ScatNet = ScatNetCNNHybrid().to(DEVICE)
weights_file = 'hybrid_all_subs.pth'
summary_file = 'hybrid_all_subs.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader_all_subs, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid (all)')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
