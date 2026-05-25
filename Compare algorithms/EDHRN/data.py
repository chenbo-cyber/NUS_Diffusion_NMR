import math
import torch
import numpy as np
import scipy.io as scio
import random
import scipy
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

path = r"EDHRN/label_spec"
matdata = scio.loadmat(path)
# matdata1 = scio.loadmat(path1)

ideal = np.array(matdata['label_spec'][:].tolist()).T
ideal1 = np.array(matdata['label_spec'][:].tolist()).T
# under = np.array(matdata1['spec_HSQC_Gb1_norm_under_fre'][:].tolist())
N1 = 170
pad_zeros = N1 - ideal.shape[1]
fn = 170
N2 = ideal.shape[0]

result_formatted = random.randint(1, 1000)  # 生成1到10000之间的随机整数
print(result_formatted)
# result_formatted = 30  # 固定采样模板
Mask = scio.loadmat('./EDHRN/0.15_stack_mask-170-768/Mask_' + str(result_formatted) + '.mat')
Mask = Mask['Mask']
Mask = Mask[:, 1].reshape(1, -1)
Mask = np.tile(Mask, (N2, 1))

# 使用 np.pad 补零
# ideal = np.pad(ideal, ((0, 0), (0, pad_zeros)), mode='constant')
# ideal1 = np.pad(ideal1, ((0, 0), (0, pad_zeros)), mode='constant')
under = np.multiply(Mask, np.fft.ifft(ideal, axis=1))
under = np.abs(np.fft.fft(under, axis=1))
# under = np.pad(under, ((0, 0), (0, pad_zeros)), mode='constant')


max_ideal = np.expand_dims(np.max(ideal, axis=1), axis=1)
ideal = ideal / max_ideal
a = np.max(ideal)

max_under = np.expand_dims(np.max(under, axis=1), axis=1)
under = under / max_under
b = np.max(under)

test_list = np.c_[under, ideal]
test_dataset =test_list

class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:]

        src_data = src_data.reshape(1, fn)
        trg_data = trg_data.reshape(1, fn)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths
