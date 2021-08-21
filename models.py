from torch.nn import Module, Linear, ReLU, Sequential, Dropout, Conv2d, BatchNorm2d, AdaptiveAvgPool2d
from kymatio.torch import Scattering2D
import torch
from time import time


class MultiScalePretrained(Module):
    def __init__(self, pretrained_model = None, n_classes = 60, device = 'cuda:0'):
        """
        Model using pretrained architechtures as the CNN feature extractor

        Parameters
        ----------
        pretrained_model: torchvision.models
            pretrained model e.g. ResNet passed to act as the CNN layers of the MultiScalePretrained model

        n_classes: int
            Number of output classes
        """
        super().__init__()

        self.n_classes = n_classes
        self.device = device

        self.model = pretrained_model.to(self.device) # send pretrained model to cpu, ensures model can be redefined after being put on the gpu
        self._get_in_features() # function to calculate the input features or similaryly output shape of the pretrained models

        # define fully connected layers for each image size
        self.fc_large = Linear(self._in_features, n_classes)
        self.fc_med = Linear(self._in_features, n_classes)
        self.fc_small = Linear(self._in_features, n_classes)


    def _get_in_features(self):
        x = torch.randn(1,3,40,40).to(self.device) # mock data
        output = self.model(x) # pass through pretraied model
        self._in_features = output.flatten().shape[0] # output shape of the pretrained model


    def forward(self, large, med, small = None):
        # pass images through pretrained network and add linear classifier to the end
        out_large = self.model(large)
        out_large = self.fc_large(out_large)

        out_med = self.model(med)
        out_med = self.fc_med(out_med)

        if small is not None:
            out_small = self.model(small)
            out_small = self.fc_small(out_small)

            out = (out_large + out_med + out_small)/3 # average the outputs of the 3 scales of images for final classification

        else:
            # out_large = torch.nn.functional.softmax(out_large)
            # out_med = torch.nn.functional.softmax(out_med)
            out = (out_large + out_med)/2

        return out




class ScatteringModel(Module):
    def __init__(self, input_size, hidden_units = None, n_classes = 60):
        """
        Variable model architechture, fusing the results of 3 deep fully connected networks for classification

        Parameters
        ----------
        input_size: int
            Number of units in the input layer of the network

        hidden_units: list or iterable of ints
            Iterable contained the number of units in the hidden layers of the network, ReLU activations are used at each layer

        n_classes: int
            Number of output classes
        """
        super().__init__()

        self.n_classes = n_classes

        if hidden_units is None: # if no hidden units passed, a linear classifier is assumed
            self.classifier_large = Sequential(Linear(input_size, n_classes))
            self.classifier_med = Sequential(Linear(input_size, n_classes))
            self.classifier_small = Sequential(Linear(input_size, n_classes))

        else:
            # lists to hold the layer and activation objects to be unpacked into a sequential object
            c_large = list()
            c_med = list()
            c_small = list()

            for unit in hidden_units:
                if unit == 'D':
                    c_large.append(Dropout(0.5))
                    c_med.append(Dropout(0.5))
                    c_small.append(Dropout(0.5))
                else:
                    # add a fully connected layer and ReLU activation to the lists
                    c_large.extend([Linear(input_size, unit), ReLU()])
                    c_med.extend([Linear(input_size, unit), ReLU()])
                    c_small.extend([Linear(input_size, unit), ReLU()])

                    input_size = unit # the output of one layer become the input to another and so needs redefining

            # add the final classification layer, the output to the number of classes
            c_large.append(Linear(input_size, n_classes))
            c_med.append(Linear(input_size, n_classes))
            c_small.append(Linear(input_size, n_classes))

            # unpack the lists to create the Sequential objects that data can be passed through
            self.classifier_large = Sequential(*c_large)
            self.classifier_med = Sequential(*c_med)
            self.classifier_small = Sequential(*c_small)

    def forward(self, large, med, small = None):

        # pass data through the classifiers
        out_large = self.classifier_large(large)
        out_med = self.classifier_med(med)

        if small is not None:
            out_small = self.classifier_small(small)
            out = (out_large + out_med + out_small)/3 # average the outputs for final classification

        else:
            # out_large = torch.nn.functional.softmax(out_large)
            # out_med = torch.nn.functional.softmax(out_med)
            out = (out_large + out_med)/2


        return out


class ScatNetCNNHybrid(Module):
    def __init__(self, J=4, L=8, shape = (128,128), n_classes = 60):
        super().__init__()
        self.J = J
        self.L = L
        self.n_classes = n_classes

        self.scattering = Scattering2D(J=J, L=L, shape = shape)

        self.K = int(3*(1 + J*L + (L**2)*J*(J-1)/2))
        # self.bn = BatchNorm2d(self.K)

        self.convs = [Conv2d(self.K, 256, kernel_size = 3, padding = 1), ReLU(),
                    Conv2d(256, 512, kernel_size = 3, padding = 1), ReLU(),
                    Conv2d(512, 512, kernel_size = 3, padding = 1), ReLU(),
                    Conv2d(512, 1024, kernel_size = 3, padding = 1), ReLU()]

        self.convs.append(AdaptiveAvgPool2d(1))

        self.convs = Sequential(*self.convs)

        self.fc = Linear(1024, n_classes)


    def forward(self, x):
        # out = self.scattering(x)
        # out = out.view(-1, self.K, 8, 8)
        out = self.convs(x)
        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = ScatNetCNNHybrid(3,8).cuda()
    x = torch.randn(128,3,128,128).cuda()
    t = time()
    for i in range(500):
        print(model(x).shape, i)
    print(time()-t)
    print(sum([i.numel() for i in model.parameters()]))
