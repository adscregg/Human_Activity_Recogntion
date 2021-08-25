from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim

from runHybrid import runModel as Hybridrun
from run import runModel

from torchvision.models import resnet101, shufflenet_v2_x1_0

from dataset_classes import imageDataset, scatteringDataset
from models import MultiScalePretrained, ScatteringModel, ScatNetCNNHybrid
from identity import Identity






BATCH_SIZE = 128
DEVICE = 'cuda'
TRAIN_SUBJECT_IDS = [1,2,3,4,5]
TEST_SUBJECT_IDS = [6,7,8,9,10]
loss_fn = CrossEntropyLoss()

def input_size(J, L):
    n = 1 + L*J + L**2 * J * (J - 1)/2
    n *= 3
    return int(n)

in_size_J4_L8 = input_size(4, 8) * 2*2
in_size_J2_L8_44 = input_size(2,8) * 6*6
in_size_J5_L8 = input_size(5, 8)

classes = 20
lin = None
shallow = [256, 512, 'D']
deep = [512, 1024, 1024, 512, 256, 'D']


# WEIGHTS_PATH = './weights/MSR_Action3D/J2_L8_44/'
# SUMMARIES_PATH = './model_summaries/MSR_Action3D/J2_L8_44/'
# scattering_dir = 'C:/Local/scattering_coeffs_J2_L8_mlp/' # note, the J2 files are not single scale
# scattering_train = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS, msr = True)
# scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS, msr = True)
# scattering_trainloader = DataLoader(scattering_train, batch_size = BATCH_SIZE, shuffle = True)
# scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)
#
# ScatNet = ScatteringModel(in_size_J2_L8_44, lin, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_lin.pth'
# summary_file = 'scattering_lin.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.01)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering lin')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
#
#
# ScatNet = ScatteringModel(in_size_J2_L8_44, shallow, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_shallow.pth'
# summary_file = 'scattering_shallow.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.01)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering shallow')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
#
#
# ScatNet = ScatteringModel(in_size_J2_L8_44, deep, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_deep.pth'
# summary_file = 'scattering_deep.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering deep')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
#
#
#
#
#
#
#
#
#
# WEIGHTS_PATH = './weights/MSR_Action3D/J4_L8/'
# SUMMARIES_PATH = './model_summaries/MSR_Action3D/J4_L8/'
# scattering_dir = 'C:/Local/scattering_coeffs_J4_L8_mlp/' # note, the J2 files are not single scale
# scattering_train = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS, msr = True)
# scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS, msr = True)
# scattering_trainloader = DataLoader(scattering_train, batch_size = BATCH_SIZE, shuffle = True)
# scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)
#
# ScatNet = ScatteringModel(in_size_J4_L8, lin, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_lin.pth'
# summary_file = 'scattering_lin.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.01)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering lin')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
#
#
# ScatNet = ScatteringModel(in_size_J4_L8, shallow, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_shallow.pth'
# summary_file = 'scattering_shallow.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.01)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering shallow')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)
#
#
# ScatNet = ScatteringModel(in_size_J4_L8, deep, n_classes = 20).to(DEVICE)
# weights_file = 'scattering_deep.pth'
# summary_file = 'scattering_deep.json'
# optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
# ScatNet_class = runModel(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
# ScatNet_class.train(epochs = 100, validate = True)
# ScatNet_class.create_model_summary('Scattering deep')
# ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
# ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)











WEIGHTS_PATH = './weights/MSR_Action3D/Hybrid_J4/'
SUMMARIES_PATH = './model_summaries/MSR_Action3D/Hybrid_J4/'
scattering_dir = 'C:/Local/scattering_coeffs_J4_L8_hybrid/' # note, the J2 files are not single scale
scattering_train = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS, msr = True, multi = False)
scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS, msr = True, multi = False)
scattering_trainloader = DataLoader(scattering_train, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)

ScatNet = ScatNetCNNHybrid(J=4, L=8, n_classes = 20).to(DEVICE)
weights_file = 'Hybrid J4_L8.pth'
summary_file = 'Hybrid J4_L8.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = Hybridrun(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid J4_L8')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)










WEIGHTS_PATH = './weights/MSR_Action3D/Hybrid_J2/'
SUMMARIES_PATH = './model_summaries/MSR_Action3D/Hybrid_J2/'
scattering_dir = 'C:/Local/scattering_coeffs_J2_L8_hybrid/' # note, the J2 files are not single scale
scattering_train = scatteringDataset(scattering_dir, TRAIN_SUBJECT_IDS, msr = True, multi = False)
scattering_test = scatteringDataset(scattering_dir, TEST_SUBJECT_IDS, msr = True, multi = False)
scattering_trainloader = DataLoader(scattering_train, batch_size = BATCH_SIZE, shuffle = True)
scattering_testloader = DataLoader(scattering_test, batch_size = BATCH_SIZE, shuffle = True)

ScatNet = ScatNetCNNHybrid(J=2, L=8, n_classes = 20).to(DEVICE)
weights_file = 'Hybrid J4_L8.pth'
summary_file = 'Hybrid J4_L8.json'
optimiser = optim.Adam(ScatNet.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.2, step_size = 25)
ScatNet_class = Hybridrun(ScatNet, DEVICE, optimiser, loss_fn, scattering_trainloader, scattering_testloader, scheduler)
ScatNet_class.train(epochs = 100, validate = True)
ScatNet_class.create_model_summary('Hybrid J4_L8')
ScatNet_class.save_model(WEIGHTS_PATH + weights_file)
ScatNet_class.save_model_summary(SUMMARIES_PATH + summary_file)










WEIGHTS_PATH = './weights/MSR_Action3D/CNNs/'
SUMMARIES_PATH = './model_summaries/MSR_Action3D/CNNs/'
image_dir = 'C:/Local/images/' # note, the J2 files are not single scale
images_train = imageDataset(image_dir, TRAIN_SUBJECT_IDS, msr = True)
images_test = imageDataset(image_dir, TEST_SUBJECT_IDS, msr = True)
images_trainloader = DataLoader(images_train, batch_size = BATCH_SIZE, shuffle = True)
images_testloader = DataLoader(images_test, batch_size = BATCH_SIZE, shuffle = True)

# ResNet = resnet101(pretrained = True)
# ResNet.fc = Identity()
# MS_CNN = MultiScalePretrained(ResNet).to(DEVICE)
# weights_file = 'ResNet.pth'
# summary_file = 'ResNet.json'
# optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
# scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
# MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader, images_testloader, scheduler)
# MS_CNN_class.train(epochs = 100, validate = True) # start training loop
# MS_CNN_class.create_model_summary('ResNet')
# MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
# MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)

ShuffleNet = shufflenet_v2_x1_0(pretrained = True)
ShuffleNet.fc = Identity()
MS_CNN = MultiScalePretrained(ShuffleNet).to(DEVICE)
weights_file = 'ShuffleNet.pth'
summary_file = 'ShuffleNet.json'
optimiser = optim.Adam(MS_CNN.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimiser, gamma = 0.1, step_size = 25)
MS_CNN_class = runModel(MS_CNN, DEVICE, optimiser, loss_fn, images_trainloader, images_testloader, scheduler)
MS_CNN_class.train(epochs = 100, validate = True)
MS_CNN_class.create_model_summary('ShuffleNet')
MS_CNN_class.save_model(WEIGHTS_PATH + weights_file)
MS_CNN_class.save_model_summary(SUMMARIES_PATH + summary_file)
