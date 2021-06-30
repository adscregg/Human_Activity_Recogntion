import warnings
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image

class scatteringDataset(Dataset):
    def __init__(self, scattering_dict = None, subjects = None):
        """
        scattering_dict: dict
            keys of the form SsssCcccPpppRrrrAaaa where  sss is the setup number, ccc is the camera ID,
            ppp is the performer (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

            values are tuples of the form (torch.Tensor, ... torch.Tensor, target)

        subject: list
            list of integers containing the performer (subject) IDs to be used in the dataset
        """

        if scattering_dict is None:
            raise ValueError('Please provide a dictionary of flattened and pooling scattering coeffs')
        if subjects is None:
            warnings.warn('No value is specified for the subjects to be used, all subjects are being used by default')
            subjects = list(range(1,41))

        self.scattering_dict = scattering_dict
        self.subjects = ['P' + f"{a:03}" for a in subjects]
        self.samples = []

        for k, v in self.scattering_dict.items():
            if k in self.subjects:
                self.samples.append(v)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class imageDataset(Dataset):
    def __init__(self, image_dir = None, subjects = None):
        """
        image_dir: string
            local or global file path to the directory containing images

        subject: list
            list of integers containing the performer (subject) IDs to be used in the dataset
        """

        if image_dir is None:
            raise ValueError('Please provide the directory of the images')

        self.image_dir = image_dir
        self.all_files = os.listdir(image_dir)

        if subjects is None:
            warnings.warn('No value is specified for the subjects to be used, all subjects are being used by default')
            subjects = list(range(1,41))

        self.subjects = ['P' + f"{a:03}" for a in subjects]

        self.accepted_files = [file for file in self.all_files if file[8:12] in self.subjects]

        self.trans100 = T.Compose([T.Resize((100,100)), T.ToTensor()])
        self.trans64 = T.Compose([T.Resize((64,64)), T.ToTensor()])
        self.trans32 = T.Compose([T.Resize((32,32)), T.ToTensor()])

    def __len__(self):
        return len(self.accepted_files)

    def __getitem__(self, idx):
        file_name = self.accepted_files[idx]
        file_path = os.path.join(self.image_dir, file_name)
        image = PIL.Image.open(file_path).convert("RGB")

        im100 = self.trans100(image)
        im64 = self.trans64(image)
        im32 = self.trans32(image)
        target = int(file_name[17:20])-1

        return im100, im64, im32, target
