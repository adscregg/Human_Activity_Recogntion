import warnings
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image

class scatteringDataset(Dataset):
    def __init__(self, scattering_dict = None, subjects = None):
        """
        Dataset class for scattering coefficients stored in a dictionary

        Parameters
        ------------
        scattering_dict: dict
            keys of the form SsssCcccPpppRrrrAaaa where  sss is the setup number, ccc is the camera ID,
            ppp is the performer (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

            values are tuples of the form (torch.Tensor, ... torch.Tensor, target)

        subject: list
            list of integers containing the performer (subject) IDs to be used in the dataset

        """
        # check if arguments have been passed and throw relevant error/warning
        if scattering_dict is None:
            raise ValueError('Please provide a dictionary of flattened and pooling scattering coeffs')
        if subjects is None:
            warnings.warn('No value is specified for the subjects to be used, all subjects are being used by default')
            subjects = list(range(1,41)) # 40 subjects in the NTu RGB+D dataset

        self.scattering_dict = scattering_dict
        self.subjects = ['P' + f"{a:03}" for a in subjects] # string form of the subject id to filter the dict keys by
        self.samples = [] # empty list, will contain samples to be accepted into the dataset

        for k, v in self.scattering_dict.items(): # loop over the key, value pairs
            if k[8:12] in self.subjects: # check if the subject id is in the list that are being accepted
                self.samples.append(v) # add sample to accepted samples of the dataset

    def __len__(self):
        return len(self.samples) # number of samples

    def __getitem__(self, idx):
        return self.samples[idx]


class imageDataset(Dataset):
    def __init__(self, image_dir = None, subjects = None, large = (100,100), med = (64,64), small = (40,40)):
        """
        Dataset class for a directory of images

        Parameters
        ------------
        image_dir: string
            local or global file path to the directory containing images

        subject: list
            list of integers containing the performer (subject) IDs to be used in the dataset

        large: tuple (N, M)
            size to reshape image to. Defaults to (100, 100)

        med: tuple (N, M)
            see `large`. Defaults to (64, 64)

        small: tuple (N, M)
            see `large`. Defaults to (40, 40)

        """

        # check if arguments have been passed and throw relevant error/warning
        if image_dir is None:
            raise ValueError('Please provide the directory of the images')
        if subjects is None:
            warnings.warn('No value is specified for the subjects to be used, all subjects are being used by default')
            subjects = list(range(1,41)) # 40 subjects in the NTU RGB+D dataset


        self.image_dir = image_dir
        self.large = large
        self.med = med
        self.small = small
        self.all_files = set(os.listdir(image_dir)).difference({'Thumbs.db'}) # set of all files excluding the Thumbs.db file if present in directory



        self.subjects = ['P' + f"{a:03}" for a in subjects] # string form of the subject ids to filter the file names by

        self.accepted_files = [file for file in self.all_files if file[8:12] in self.subjects] # file names that are accepted as part of the dataset

        # Define transformation that will resize the images to (N, M) and convert them to torch.Tensor objects
        self.trans_large = T.Compose([T.Resize(self.large), T.ToTensor()])
        self.trans_med = T.Compose([T.Resize(self.med), T.ToTensor()])
        self.trans_small = T.Compose([T.Resize(self.small), T.ToTensor()])

    def __len__(self):
        return len(self.accepted_files) # number of files in the dataset

    def __getitem__(self, idx):
        file_name = self.accepted_files[idx]
        file_path = os.path.join(self.image_dir, file_name) # full path to image
        image = Image.open(file_path).convert("RGB") # open image and convert to RGB to ensure 3 channels

        # Apply transformations to the images
        im_large = self.trans_large(image)
        im_med = self.trans_med(image)
        im_small = self.trans_small(image)

        target = int(file_name[17:20])-1 # target id, has -1 so the class ids start at 0, easy for pytorch to handle

        return im_large, im_med, im_small, target
