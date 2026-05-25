import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random
import scipy.io as scio
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
import h5py

class MyDataset(dataset.Dataset):

    def __init__(self, num_samples, split='train'):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        NOISE_input, GT_label,  factor_matrix= self.__gen_signal__(idx)

        max_amp = np.max(np.abs(NOISE_input))
        NOISE_input = np.abs(NOISE_input) / (max_amp)
        GT_label = np.abs(GT_label) / (max_amp * 8)
        return NOISE_input, GT_label, factor_matrix

    def __gen_signal__(self, idx):
        dim = 1
        N = 256
        max_J = 12

        J = np.random.randint(1, max_J + 1, size=(dim, 1))  # Random number of harmonics
        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel() - 1] = 1  # Create a mask for harmonics
        mask = np.cumsum(mask, axis=1)

        ph = np.random.uniform(0.0, 0.2 * np.pi, size=(dim, max_J))  # Random phase
        A = np.random.uniform(0.05, 1.0, size=(dim, max_J))  # Random amplitude
        w = np.random.uniform(0.01, 0.99, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time

        t = np.arange(N)
        A = np.multiply(A, mask)
        # Generate the 1D FID signal

        x = (A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t / sgm[..., None]) *
             np.exp(1j * 2 * np.pi * w[..., None] * t))
        clean_xn = np.sum(x, axis=1)

        # Add noise to the FID signal
        noise_scale = 1e-4
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=N)
        xx = noise + clean_xn

        threshold = 3
        temp = np.zeros((dim, max_J, N))

        for i in range(dim):
            freq_indices1 = (
                        np.abs(2 * np.pi * w[i, :] - (np.linspace(0, 1, N) * 2 * np.pi)[:, None]) <= threshold / sgm[i,
                                                                                                                 :])
            freq_indices1 = freq_indices1.T
            temp[i, freq_indices1] = 1
        A1 = np.multiply(np.ones((dim, max_J)), mask)
        temp1 = A1[0][:, np.newaxis, np.newaxis] * temp[0][:, :, np.newaxis]
        factor_matrix = np.sum(temp1, axis=0)
        factor_matrix[factor_matrix > 1] = 1
        factor_matrix = factor_matrix.T

        # 采样率固定
        # result_formatted = random.randint(1, 10000)  # 生成1到10000之间的随机整数
        result_formatted = 30  # 固定采样模板
        Mask = scio.loadmat('./0.2_stack_mask-256-1024/Mask_' + str(result_formatted) + '.mat')
        Mask = Mask['Mask']
        U = Mask[:, 1].reshape(1, -1)

        NOISE_FID = np.multiply(U, xx)
        NOISE_input = np.fft.fft(NOISE_FID, axis=1)
        GT_label = np.fft.fft(xx, axis=1)

        return NOISE_input, GT_label, factor_matrix


def create_dataset(phase):
    '''create dataset'''

    if phase == 'train':
        dataset = MyDataset(
                    num_samples=40000,
                    split=phase,
                                    )
    elif phase == 'val':
        dataset = MyDataset(num_samples=5000,
                    split=phase,)
    return dataset

def create_dataloader(dataset, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False)


