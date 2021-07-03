from torch.nn import Module, Linear, ReLU, Sequential
import torch


class MultiScalePretrained(Module):
    def __init__(self, pretrained_model = None, n_classes = 60):
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

        self.model = pretrained_model.cpu() # send pretrained model to cpu, ensures model can be redefined after being put on the gpu
        self._get_in_features() # function to calculate the input features or similaryly output shape of the pretrained models

        # define fully connected layers for each image size
        self.fc_large = Linear(self._in_features, n_classes)
        self.fc_med = Linear(self._in_features, n_classes)
        self.fc_small = Linear(self._in_features, n_classes)


    def _get_in_features(self):
        x = torch.randn(1,3,40,40) # mock data
        output = self.model(x) # pass through pretraied model
        self._in_features = output.flatten().shape[0] # output shape of the pretrained model


    def forward(self, large, med, small):
        # pass images through pretrained network and add linear classifier to the end
        out_large = self.model(large)
        out_large = self.fc_large(out_large)

        out_med = self.model(med)
        out_med = self.fc_med(out_med)

        out_small = self.model(small)
        out_small = self.fc_small(out_small)

        out = (out_large + out_med + out_small)/3 # average the outputs of the 3 scales of images for final classification

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

    def forward(self, large, med, small):

        # pass data through the classifiers
        out_large = self.classifier_large(large)
        out_med = self.classifier_med(med)
        out_small = self.classifier_small(small)

        out = (out_large + out_med + out_small)/3 # average the outputs for final classification

        return out
