import torch
from torch.utils import data
from torchvision.transforms import transforms

import numpy as np
import h5py
import os
import math
import matplotlib.pyplot as plt

from skimage.transform import resize
from scipy import interpolate

class PhC2D(data.Dataset):

    def __init__(self, path_to_h5_dir, percent_labeled=0.1, testsize=1500,
                split='train_lb'):
        filename = 'bandgap_5classes_4.h5'
        # index_file = f'./phc_npy/phc_{percent_labeled}_lab_new.npy'
        # ulb_index_file = f'./phc_npy/phc_{percent_labeled}_unlab_new.npy'
        index_file = f'./phc_npy/phc_100percent_lb.npy'
        ulb_index_file = f'./phc_npy/phc_100percent_ulb.npy'
        
        test_file = f'./phc_npy/phc_{percent_labeled}_test_new.npy'
        
        if split == 'train_lb':
            indlist = np.load(index_file)
            print("loaded: ", index_file)
        elif split == 'train_ulb':
            indlist = np.load(ulb_index_file) # all test files for diff fraction should be the same
            print("loaded: ", ulb_index_file)
        elif split == 'test':
            indlist = np.load(test_file) # all test files for diff fraction should be the same
            print("loaded: ", test_file)

        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []

        with h5py.File(os.path.join(path_to_h5_dir, filename), 'r') as f:
            for memb in indlist:
                input = f['input_uc/'+str(memb)][()]
                y = f['class/'+str(memb)][()]
                self.x_data.append(input)
                self.y_data.append(y)

        # normalize x data
        self.x_data = (np.array(self.x_data).astype('float32')-1) / 19 # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('long')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        :return: random sample for a task
        """
        ## input always first element in tuple and output always second element
        return self.x_data[index], self.y_data[index]

def get_phc(args, root):
    train_labeled_dataset = PhC2D(root, args.labeled_perc, split='train_lb')
    train_unlabeled_dataset = PhC2D(root, args.labeled_perc, split='train_ulb')
    test_dataset = PhC2D(root, args.labeled_perc, split='test')
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
