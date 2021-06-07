import numpy as np
import os

data_loc = "data/MSRAction3D/MSRAction3DSkeleton(20joints)/"

def toImage(skel):
    skeleton = np.loadtxt(skel)
    skeleton = skeleton.reshape((20, -1, 4), order='F')
    
    X = skeleton[:, :, 0]
    Z = 400 - skeleton[:, :, 1]
    Y = skeleton[:, :, 2]/4
    
    im = np.array([X, Y, Z])
    
    return im

def transformImage(im):
    pass
