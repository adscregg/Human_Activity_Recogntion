import torch
import os
from tqdm import tqdm

if __name__ == '__main__':
    _64bit_path = './data/NTU_RGB+D/scattering_coeffs_64bit/'
    save = './data/NTU_RGB+D/scattering_coeffs_32bit/'
    files = os.listdir(_64bit_path)
    for file in tqdm(files):
        l,m,s,t = torch.load(_64bit_path + file)
        l,m,s = l.type(torch.float32),m.type(torch.float32),s.type(torch.float32)
        torch.save((l,m,s,t), save + file)
