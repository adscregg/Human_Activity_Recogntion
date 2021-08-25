import torch
import inspect
from tqdm import tqdm
from time import time
import json
import numpy as np

from torch.cuda.amp import autocast, GradScaler



class runModel:
    def __init__(self, model, device, optimiser, loss_fn, train_loader, test_loader, scheduler = None, use_amp = True):
        """
        Class for training and evaluating deep learning models

        Parameters
        ----------
        model: torch.nn.Module
            PyTorch model

        device: str
            Either 'cpu' or 'cuda' (specify 'cuda:n' if more than one cuda enabled GPU device is available)

        optimiser: torch.optim
            Optimiser for updating the weights of the model

        loss_fn: torch.nn or torch.nn.functional
            Function to calculate the loss of the outputs of the model

        train_loader: torch.utils.data.DataLoader
            Generator object of the training samples for training

        test_loader: torch.utils.data.DataLoader
            Generator object of the training samples for testing

        scheduler: torch.optimi.lr_scheduler
            Reduce the learning rate during training to avoid overfitting, only takes affect if validation cycle is done (validate = True)

        use_amp: bool
            whether to use mixed precision during training
        """
        self.model = model
        self.device = device
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

        self.n_parameters = sum([i.numel() for i in model.parameters()])
        self.num_test_samples = len(self.test_loader.dataset)
        self.num_train_samples = len(self.train_loader.dataset)

        self.train_accuracy = None
        self.test_accuracy = None
        self.train_acc_history = list()
        self.test_acc_history = list()

        self.train_loss = None
        self.test_loss = None
        self.train_loss_history = list()
        self.test_loss_history = list()

        self.lr_history = list()
        self.average_time_per_epoch = None
        self.num_epochs = None
        self._epoch_times = list()

        self.confusion_matrix = None



    def train(self, epochs, validate = False):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            Number of loops over the dataset

        validate: bool
            Whether to evaluate the model on the test set throughout the training process.
        """
        # initialise lists to track the metrics on the training set throughout training
        self.train_loss_history = list()
        self.train_acc_history = list()
        self.lr_history = list()
        self._epoch_times = list()
        self.num_epochs = epochs

        # self.min_test_loss = np.inf


        # if validate is set to True, track the metrics on the test set throughout training
        if validate:
            self.test_loss_history = list()
            self.test_acc_history = list()


        progress = tqdm(range(epochs), desc = 'Training')
        progress.set_postfix(train_accuracy = self.train_accuracy, train_loss = self.train_loss, test_acc = self.test_accuracy, test_loss = self.test_loss)

        for epoch in progress:

            self.model.train() # put the model into train mode, includes layers, e.g. Dropout, that are only used for training
            epoch_loss = 0 # reset the stats for each epoch
            n_correct = 0
            start = time()

            for value, target in self.train_loader:
                # load samples and targets onto the specified device (cpu or gpu)
                value, target = value.to(self.device), target.to(self.device)

                with autocast(enabled=self.use_amp):
                    output = self.model(value) # run the model
                    loss = self.loss_fn(output, target) # calculate the loss

                epoch_loss += loss*target.size(0) # loss is the average for all sample in the batch, total loss is avg*batch_size
                pred = torch.argmax(output, dim = 1) # make the prediction by taking the largest output value for each sample
                n_correct += (pred == target).float().sum()#.item() # calculate how many are correct

                # loss.backward() # backward pass through the model, or backpropagation
                self.scaler.scale(loss).backward()

                # self.optimiser.step() # update the weights
                self.scaler.step(self.optimiser)
                self.scaler.update()

                self.optimiser.zero_grad(set_to_none=True) # ensures the gradient does not accumalate in the optimiser

            epoch_time = time() - start
            self._epoch_times.append(epoch_time)

            self.train_loss = epoch_loss.item() / self.num_train_samples
            self.train_accuracy = n_correct.long().item() / self.num_train_samples

            self.train_loss_history.append(self.train_loss) # average loss for all samples in the epoch
            self.train_acc_history.append(self.train_accuracy) # accuracy score, between 0 and 1


            if validate:
                self.test() # run the test loop

                # append the metrics on the test set
                self.test_loss_history.append(self.test_loss)
                self.test_acc_history.append(self.test_accuracy)

                if self.scheduler is not None:
                    if 'metrics' in inspect.getfullargspec(self.scheduler.step)[0]:
                        self.scheduler.step(self.test_loss)
                    else:
                        self.scheduler.step()

            else:
                self.test_loss_history.append(None)
                self.test_acc_history.append(None)


            progress.set_postfix(train_accuracy = self.train_accuracy, train_loss = self.train_loss, test_acc = self.test_accuracy, test_loss = self.test_loss) # update progress bar outputs


            self.lr_history.append(self.optimiser.param_groups[0]['lr'])

        self.average_time_per_epoch = sum(self._epoch_times)/len(self._epoch_times)



    def test(self):
        """
        Test the model performance on the test set

        Parameters
        ----------
        None
        """
        self.model.eval() # put the model into evaluate mode, removes layers, e.g. Dropout, that are only used for training
        self.test_loss = 0
        n_correct = 0
        # progress = tqdm(self.test_loader, desc = 'Testing', leave=False, position = 0)
        self.confusion_matrix = torch.zeros(self.model.n_classes, self.model.n_classes)
        with torch.no_grad(): # let PyTorch know that no gradient information is needed as no training is being done, faster evaluation
            for value, target in self.test_loader:
                value, target = value.to(self.device), target.to(self.device)

                output = self.model(value)
                loss = self.loss_fn(output, target)

                self.test_loss += loss.item()*target.size(0)
                pred = torch.argmax(output, dim = 1)
                n_correct += (pred == target).float().sum().item()

                for t, p in zip(target.flatten(), pred.flatten()):
                    self.confusion_matrix[t.long(), p.long()] += 1

            self.test_loss /= self.num_test_samples
            self.test_accuracy = n_correct / self.num_test_samples
            self.confusion_matrix = self.confusion_matrix.numpy().tolist()



    def save_model(self, file_path):
        """
        Save the weights of the model

        Parameters
        ----------
        file_path: str
            Save location of the model weights. Common PyTorch convention is to save models using either a .pt or .pth file extension
        """
        torch.save(self.model.state_dict(), file_path)



    def load_weights(self, file_path):
        """
        Save the weights of the model

        Parameters
        ----------
        file_path: str
            Location of the saved model weights
        """
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval() # default to evaluation mode after loading the weights



    def create_model_summary(self, name):

        self.model_summary = {
        'name': name,
        'num parameters': self.n_parameters,

        'num train samples': self.num_train_samples,
        'num test samples': self.num_test_samples,
        'num epochs': self.num_epochs,

        'train accuracy': self.train_accuracy,
        'train loss': self.train_loss,
        'test accuracy': self.test_accuracy,
        'test loss': self.test_loss,

        'train accuracy history': self.train_acc_history,
        'train loss history': self.train_loss_history,
        'test accuracy history': self.test_acc_history,
        'test loss history': self.test_loss_history,

        'lr history': self.lr_history,
        'avg time per epoch': self.average_time_per_epoch,
        'total training time': sum(self._epoch_times),

        'confusion matrix': self.confusion_matrix
        }



    def save_model_summary(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.model_summary, fp, indent = 4)
