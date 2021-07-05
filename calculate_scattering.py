import os
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
from kymatio.torch import Scattering2D
from tqdm import tqdm


def preprocessScatteringCoeffs(image_dir, save_dir, J, L= 8, batch_size = 64, large = (100,100), med = (64,64), small = (40,40)):
    """
    Calculates the pooled and flattened scattering coefficients for all images in a directory

    Parameters
    -----------
    image_dir: string
        File path to directory of images

    save_dir: string
        File path to save the calculated coefficients

    J: int
        Log-2 of the scattering scale

    L: int
        Number of angles used for the wavelet transform. Defaults to 8

    batch_size: int
        Number of samples to calculate scattering coefficients of in a single pass. Defaults to 64

    large: tuple (N, M)
        size to reshape image to. Defaults to (100, 100)

    med: tuple (N, M)
        see `large`. Defaults to (64, 64)

    small: tuple (N, M)
        see `large`. Defaults to (40, 40)


    Returns
    ---------
    dict
        Dictionary object with values as the a tuple of pooled and flattened scattering coefficients for each sample

    """
    files = set(os.listdir(image_dir)).difference({'Thumbs.db'}) # create a set of files, removing the unwanted Thumbs.db file if present
    # already_calc = set(os.listdir(save_dir))


    # Scattering network initialisation and moving to GPU
    S_large = Scattering2D(J = J, L = L, shape = large).cuda()
    S_med = Scattering2D(J = J, L = L, shape = med).cuda()
    S_small = Scattering2D(J = J, L = L, shape = small).cuda()


    # Resize an image to size (N,M) and convert it to a torch.Tensor object
    trans_large = T.Compose([T.Resize(large), T.ToTensor()])
    trans_med = T.Compose([T.Resize(med), T.ToTensor()])
    trans_small = T.Compose([T.Resize(small), T.ToTensor()])

    in_stack = list() # initialise list, will hold the file names of the tensors in the stack


    # initialise empty stacks
    stack_large = torch.Tensor()
    stack_med = torch.Tensor()
    stack_small = torch.Tensor()

    scattering_dict = dict() # initialise empty dictionary, will hold the values of the scattering coefficients

    Pool = nn.AdaptiveAvgPool2d(1) # global average pooling

    i = 0 # dummy counter variable
    for file in tqdm(files): # loop over each file in the directory
        f = file[:-4]
        # if f + '.pth' in already_calc:
        #     # print(f'{f} already exists!')
        #     pass

        path = os.path.join(image_dir, file) # path to the image itself
        image = Image.open(path).convert("RGB") # read in the image and convert to RGB to ensure 3 channels


        # add a leading 0 to represent this is 1 item in a batch
        im_large = trans_large(image).unsqueeze(0)
        im_med = trans_med(image).unsqueeze(0)
        im_small = trans_small(image).unsqueeze(0)


        # stack the images so a batch is created
        stack_large = torch.cat((stack_large, im_large))
        stack_med = torch.cat((stack_med, im_med))
        stack_small = torch.cat((stack_small, im_small))

        in_stack.append(f) # add the file name to the list, need to keep track so can create dictiory with these keys

        if len(in_stack) == batch_size or i == len(files) - 1: # check if there are enough images in the batch, or have reached the end of the files

            # move stacks to GPU
            stack_large = stack_large.cuda()
            stack_med = stack_med.cuda()
            stack_small = stack_small.cuda()

            # calculate coeffs --> global pooling --> remove extra dims --> flatten --> move back to cpu to free up space
            coeffs_large = Pool(S_large(stack_large)).squeeze().flatten(start_dim = 1).cpu()
            coeffs_med = Pool(S_med(stack_med)).squeeze().flatten(start_dim = 1).cpu()
            coeffs_small = Pool(S_small(stack_small)).squeeze().flatten(start_dim = 1).cpu()

            for j, name in enumerate(in_stack):
                # target value has -1 so the classes start at 0 rather than 1, better for pytorch networks to handle
                # scattering_dict[in_stack[j]] = (coeffs_large[j], coeffs_med[j], coeffs_small[j], int(name[-3:])-1) # add the flattend coeffs and target to the dictionary with the key as the file name without the extension
                torch.save((coeffs_large[j].type(torch.float16), coeffs_med[j].type(torch.float16), coeffs_small[j].type(torch.float16), int(name[-3:])-1), save_dir[0] + name + '.pth')
                torch.save((coeffs_large[j].type(torch.float64), coeffs_med[j].type(torch.float64), coeffs_small[j].type(torch.float64), int(name[-3:])-1), save_dir[1] + name + '.pth')
            # reset the stacks and lists
            in_stack = list()
            stack_large = torch.Tensor()
            stack_med = torch.Tensor()
            stack_small = torch.Tensor()

        i += 1

    # return scattering_dict



if __name__ == '__main__':
    image_dir = './data/NTU_RGB+D/transformed_images'
    save_dir = ['./data/NTU_RGB+D/scattering_coeffs_16bit/', './data/NTU_RGB+D/scattering_coeffs_64bit/']
    preprocessScatteringCoeffs(image_dir, save_dir, J = 4, batch_size = 128)
