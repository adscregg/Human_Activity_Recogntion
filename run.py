import torch

class runModel:
    def __init__(self, model, device, optimiser, loss_fn, train_loader, test_loader, scheduler = None):
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

        scheduler (optional): torch.optim.lr_scheduler
            Reduce the learning rate during training. Defaults to None
        """
        self.model = model
        self.device = device
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

        self.n_parameters = sum([i.numel() for i in model.parameters()])
        self.num_test_samples = len(self.test_loader.dataset)
        self.num_train_samples = len(self.train_loader.dataset)

    def train(self, epochs, validate = False):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            Number of loops over the dataset

        validate: bool
            Whether to evaluate the model on the test set throughout the training process
        """
        # initialise lists to track the metrics on the training set throughout training
        self.train_loss_history = list()
        self.train_acc_history = list()

        # if validate is set to True, track the metrics on the test set throughout training
        if validate:
            self.test_loss_history = list()
            self.test_acc_history = list()

        for _ in range(epochs):
            self.model.train() # put the model into train mode, includes layers, e.g. Dropout, that are only used for training
            epoch_loss = 0 # reset the stats for each epoch
            n_correct = 0

            for large, med, small, target in self.train_loader:
                # load samples and targets onto the specified device (cpu or gpu)
                large, med, small, target = large.to(self.device), med.to(self.device), small.to(self.device), target.to(self.device)

                self.optimiser.zero_grad() # ensures the gradient does not accumalate in the optimiser

                output = self.model(large, med, small) # run the model
                loss = self.loss_fn(output, target) # calculate the loss

                epoch_loss += loss.item()*target.size(0) # loss is the average for all sample in the batch, total loss is avg*batch_size
                pred = torch.argmax(output, dim = 1) # make the prediction by taking the largest output value for each sample
                n_correct += (pred == target).float().sum() # calculate how many are correct

                loss.backward() # backward pass through the model, or backpropagation
                self.optimiser.step() # update the weights

            self.train_loss_history.append(epoch_loss / self.num_train_samples) # average loss for all samples in the epoch
            self.train_acc_history.append(n_correct / self.num_train_samples) # accuracy score, between 0 and 1

            if self.scheduler is not None:
                self.scheduler.step() # update the scheduler

            if validate:
                self.test() # run the test loop
                # append the metrics on the test set
                self.test_loss_history.append(self.test_loss)
                self.test_acc_history.append(self.test_accuracy)

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

        with torch.no_grad(): # let PyTorch know that no gradient information is needed as no training is being done, faster evaluation
            for large, med, small, target in self.test_loader:
                large, med, small, target = large.to(self.device), med.to(self.device), small.to(self.device), target.to(self.device)

                output = self.model(large, med, small)
                loss = self.loss_fn(output, target)

                self.test_loss += loss.item()*target.size(0)
                pred = torch.argmax(output, dim = 1)
                n_correct += (pred == target).float().sum()

            self.test_loss /= self.num_test_samples
            self.test_accuracy = n_correct / self.num_test_samples

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
