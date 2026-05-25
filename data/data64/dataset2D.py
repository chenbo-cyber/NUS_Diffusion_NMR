import os
import torch
import logging
import math
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset
import scipy.interpolate

class NUSNMRDataset(Dataset):
    def __init__(self, num_samples, split='train', need_LR=False):
        super(NUSNMRDataset, self).__init__()
        self.num_samples = num_samples
        self.split = split
        self.mul = 10000
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        NUS_FID, NOISE_input, GT_label, factor_matrix = self.__gen_signal__(idx)
        # power_of_10 = int(np.log10(np.max([np.real(NOISE_input), np.imag(NOISE_input)]))) + 1 if np.max([np.real(NOISE_input), np.imag(NOISE_input)]) != 0 else 1
        # max_amp = 10 ** power_of_10
        max_amp = np.max(np.abs(NOISE_input))

        # NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float())
        NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float()) / (max_amp * 12.5)
        NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / (max_amp)
        GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / (max_amp * 12.5)
        factor_matrix = torch.from_numpy(factor_matrix).float()

        LR = NUS_FID.unsqueeze(0)
        SR = torch.cat((NOISE_input.real.unsqueeze(0), NOISE_input.imag.unsqueeze(0)), dim=0)
        HR = torch.cat((GT_label.real.unsqueeze(0), GT_label.imag.unsqueeze(0)), dim=0)
        WR = factor_matrix.unsqueeze(0)

        return {'LR': LR, 'HR': HR, 'SR': SR, 'Index': idx, 'WR': WR}


    def __gen_signal__(self, idx):
        # Define simulation parameters
        dim = 2

        N = 64
        N1 = 64  # 第一个维度大小
        N2 = 64  # 第二个维度大小
        max_J = 20
        # max_J = random.randint(350, 550)

        # Generate FID signals
        J = np.random.randint(5, random.randint(7, 14), size=(dim, 1))  # Random number of harmonics
        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel() - 1] = 1  # TODO
        mask = np.cumsum(mask, axis=1)

        ph = np.random.uniform(0.0, 0.2 * np.pi, size=(dim, max_J))  # Random phase  # TODO
        A = np.random.uniform(0.02, 1.0, size=(dim, max_J))  # Random amplitude
        w = np.random.uniform(0.05, 0.95, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time

        t1 = np.arange(N1)  # Time axis
        t2 = np.arange(N2)  # Time axis

        A = np.multiply(A, mask)

        x1 = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t1 / sgm[..., None]) * np.exp(
            1j * 2 * np.pi * w[..., None] * t1)
        x2 = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t2 / sgm[..., None]) * np.exp(
            1j * 2 * np.pi * w[..., None] * t2)
        # xn_unit = np.matmul(x1[0][:, :, np.newaxis], x2[0][:, np.newaxis])
        xn_unit = np.matmul(x1[0][:, :, np.newaxis], x2[1][:, np.newaxis])
        clean_xn = np.sum(xn_unit, axis=0)

        # Add noise to FID signals
        noise_scale = 1e-4
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(N1, N2))
        xx = noise + clean_xn

        def generate_noise(snr, max_val, N1, N2):
            std_noise = max_val / (2 * snr)
            noise_real = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
            noise_imag = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
            noise_out = noise_real + 1j * noise_imag
            return noise_out
        # xx_noise = xx + noise_xn

        # def generate_noise(snr, max_val, N1, N2):
        #     std_noise = max_val / (2 * snr)
        #     noise_real = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
        #     noise_imag = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
        #     noise_out = noise_real + 1j * noise_imag
        #     return noise_out
        #
        # xx_noise = xx + generate_noise(np.random.uniform(40, 100), np.max(xx), N1, N2)

        # # 掩码因子改成0和1
        # factor_matrix = np.zeros((N1, N2))
        # for j in range(max_J):
        #     if j > (max(J[0], J[1]) - 2):
        #         pos1 = np.around(w[0, j] * N1)
        #         width1 = np.around(N1 / sgm[0, j])
        #         pos2 = np.around(w[1, j] * N2)
        #         width2 = np.around(N2 / sgm[1, j])
        #         factor_matrix[math.ceil(pos1 - width1 // 2):math.floor(pos1 + width1 // 2),
        #         math.ceil(pos2 - width2 // 2):math.floor(pos2 + width2 // 2)] = 1

        # Control the range of peak, can be adjusted
        threshold = 3
        # Create binary masks for the two dimensions
        temp1 = np.zeros((dim, max_J, N1))
        temp2 = np.zeros((dim, max_J, N2))

        # 减少时间复杂度
        for i in range(dim):
            freq_indices1 = (np.abs(2 * np.pi * w[i, :] - (np.linspace(0, 1, N1) * 2 * np.pi)[:, None])
                             <= threshold / sgm[i, :])
            freq_indices1 = freq_indices1.T
            temp1[i, freq_indices1] = 1
            freq_indices2 = np.abs(
                2 * np.pi * w[i, :] - (np.linspace(0, 1, N2) * 2 * np.pi)[:, None]) <= threshold / sgm[i, :]
            freq_indices2 = freq_indices2.T
            temp2[i, freq_indices2] = 1

        # Combine binary peak presence masks across dimensions
        A1 = np.multiply(np.ones((dim, max_J)), mask)
        temp_l1l2 = np.matmul(A1[0][:, np.newaxis, np.newaxis] * temp1[0][:, :, np.newaxis],
                              A1[1][:, np.newaxis, np.newaxis] * temp2[1][:, np.newaxis])
        factor_matrix = np.sum(temp_l1l2, axis=0)
        factor_matrix[factor_matrix > 1] = 1

        # 采样率固定
        result_formatted = random.randint(1, 10000)  # 生成1到10000之间的随机整数
        Mask = scio.loadmat('./data2D/0.08_2D_mask-64-64/Mask_' + str(result_formatted) + '.mat')['Mask']
        U = Mask

        NUS_FID = np.multiply(U, xx)
        NUS_FID_pad = np.pad(NUS_FID, ((0, (N - N1)), (0, (N - N2))), mode='constant', constant_values=0)

        NOISE_FID = np.multiply(U, xx)
        NOISE_FID_pad = np.pad(NOISE_FID, ((0, (N - N1)), (0, (N - N2))), mode='constant', constant_values=0)
        NOISE_input = np.fft.fft2(NOISE_FID_pad)
        GT_label = np.fft.fft2(xx)

        return NUS_FID_pad, NOISE_input, GT_label, factor_matrix

             
def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    num_samples = dataset_opt['data_len']

    dataset = NUSNMRDataset(
                num_samples=num_samples,
                split=phase,
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
