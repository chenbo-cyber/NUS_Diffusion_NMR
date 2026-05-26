import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from scipy.io import savemat


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_contour(nus_spec, recon_spec, label_spec, save_path):
    fig = plt.figure(figsize=(35, 10))
    contour_levels = 80

    ax = fig.add_subplot(131)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(132)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(133)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.savefig(save_path)
    # np.save('./data/nus_spec.npy', nus_spec)
    # np.save('./data/recon_spec.npy', recon_spec)
    # np.save('./data/label_spec.npy', label_spec)
    savemat('./data/nus_spec.mat', {'nus_spec': nus_spec})
    savemat('./data/recon_spec.mat', {'recon_spec': recon_spec})
    savemat('./data/label_spec.mat', {'label_spec': label_spec})

def save_contour2(nus_spec, recon_spec, label_spec, sr_spec, save_path):
    fig = plt.figure(figsize=(42, 10))
    contour_levels = 10

    ax = fig.add_subplot(131)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, linewidths=2, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(132)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, linewidths=2, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(133)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, linewidths=2, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(save_path)

def save_contour_auto(nus_spec, recon_spec, label_spec, sr_spec, sequence, save_path):
    fig = plt.figure(figsize=(42, 10))
    contour_levels = 80

    ax = fig.add_subplot(141)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(142)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(143)
    contour_input = ax.contour(np.abs(sr_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of sr_spec')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(144)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.savefig(save_path)
    recon_name = f'recon_spec{sequence}'
    file_path_recon = f'./data/ablationexperiments/concat/{recon_name}.mat'
    savemat(file_path_recon, {'recon_spec': recon_spec})
    label_name = f'label_spec{sequence}'
    file_path_label = f'./data/ablationexperiments/concat/{label_name}.mat'
    savemat(file_path_label, {'label_spec': label_spec})


def save_contour2D(nus_spec, recon_spec, label_spec, sr_spec, sequence, save_path, dataset_name):
    fig = plt.figure(figsize=(42, 10))
    contour_levels = 80

    ax = fig.add_subplot(141)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(142)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(143)
    contour_input = ax.contour(np.abs(sr_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of sr_spec')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(144)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.savefig(save_path)
    plt.close(fig)

    os.makedirs(f'./data/{dataset_name}_recon', exist_ok=True)
    recon_name = f'recon_spec{sequence}'
    file_path_recon = f'./data/{dataset_name}_recon/{recon_name}.mat'
    savemat(file_path_recon, {'recon_spec': recon_spec})
    label_name = f'label_spec{sequence}'
    file_path_label = f'./data/{dataset_name}_recon/{label_name}.mat'
    savemat(file_path_label, {'label_spec': label_spec})
    # plt.show()


def save_contour_test(nus_spec, recon_spec1, recon_spec2, recon_spec3, recon_spec, label_spec, save_path):
    fig = plt.figure(figsize=(30, 15))
    contour_levels = 20

    ax = fig.add_subplot(231)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(232)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(233)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')

    ax = fig.add_subplot(234)
    contour_input = ax.contour(np.abs(recon_spec1), levels=3, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result1')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(235)
    contour_input = ax.contour(np.abs(recon_spec2), levels=3, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(236)
    contour_input = ax.contour(np.abs(recon_spec3), levels=6, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result3')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.savefig(save_path)
    # np.save('nus_spec.npy', nus_spec)
    # np.save('recon_spec.npy', recon_spec)
    # np.save('label_spec.npy', label_spec)

def plot_contour(nus_spec, recon_spec, label_spec, fig_num):
    fig = plt.figure(fig_num, figsize=(15, 4))
    contour_levels = 50
    ax = fig.add_subplot(131)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(132)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(133)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')

def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_mse(hr, sr):
    return np.mean((hr - sr)**2)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
