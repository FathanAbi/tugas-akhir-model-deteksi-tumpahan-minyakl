from scipy import io, misc

import numpy as np
import os
import spectral
from sklearn import preprocessing

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    else:
        raise ValueError("Unknown file format: {}".format(ext
        ))

class HSI():
    def __init__(self, path):
        self.file = open_file(path)
       
       
        self.img = self.file["img"]
        self.gt = self.file["map"]

        name = path.split("/")[-1]
        self.name = name.split(".")[0] 



if __name__ == '__main__':
    hsi = HSI('Hyperspectral oil spill detection datasets/GM01.mat')
    # print(hsi.img.shape)
    # print(hsi.gt.shape)
    print(hsi.name)
