import math
import random
import torch
import numpy as np
import scipy.io as scio
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
import h5py
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'

path = r"Densenet/label_spec"
matdata = scio.loadmat(path)

ideal = np.array(matdata['label_spec'][:].tolist()).T
ideal1 = np.array(matdata['label_spec'][:].tolist()).T
timedata = 17
timedata = np.array(timedata)

N1 = 170
pad_zeros = N1 - ideal.shape[1]
fn = 170
N2 = ideal.shape[0]

# 采样模板
result_formatted = random.randint(1, 1000)
# result_formatted = 985
print(result_formatted)
Mask = scio.loadmat('./Densenet/0.15_stack_mask-170-768/Mask_' + str(result_formatted) + '.mat')
Mask = Mask['Mask']
U = Mask[:, 1].reshape(1, -1)

under = np.multiply(np.fft.ifft(ideal, axis=1), U)
ideal_real = np.real(under)
ideal_imag = np.imag(under)
under = np.abs(np.fft.fft(under, axis=1))

max_ideal = np.expand_dims(np.max(ideal, axis=1), axis=1)
ideal = (ideal / max_ideal)
a = np.max(ideal)
max_under = np.expand_dims(np.max(under, axis=1), axis=1)
under = (under / max_under)
b = np.max(under)

test_list = np.c_[under, ideal, ideal_real, ideal_imag]
test_dataset = test_list


class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:fn * 2]
        time_data_real = data[fn * 2:fn * 3]
        time_data_imag = data[fn * 3:]
        return src_data, trg_data, time_data_real, time_data_imag

    def __len__(self):
        return self.data_lengths