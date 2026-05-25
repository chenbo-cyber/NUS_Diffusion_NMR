import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'
import torch
import numpy as np
from data import *
from model import *
import matplotlib
import matplotlib.pyplot as plt


def getLevels(min, fac, num):
    return np.array([min*(fac**i) for i in range(num)])


def plot_contour(ax, ft_outer, col='viridis', lvl=None, invert=False):

    if lvl is None:
        lvl = getLevels(np.max(ft_outer.numpy()) * 0.035, 1.2, 22)

    plt.contour(ft_outer, levels=8, cmap=col)



def net_test_2d(device=None):
    device=device
    test_data = MyDataset(test_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)
    net.eval()


    for step, (src_data, trg_data) in enumerate(test_loader):
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = src_data.to(device)
        trg_data = trg_data.to(device)
        output = net(src_data)
        output = output.cpu()
        total_output[:, step, :] = output.detach()
        out = total_output
    out = out.cpu()
    out = out.detach().reshape(N2, N1).numpy()
    out1d = np.transpose(out)
    out = np.transpose(out*max_under)
    recon = torch.tensor(out)

    fig = plt.figure(figsize=(25, 10))
    contour_levels = 10

    ax = fig.add_subplot(121)
    contour_input = ax.contour(recon[:256-pad_zeros, :], levels=8, cmap='viridis')
    ax.set_title('Contour Plot of recon')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(122)
    contour_input = ax.contour(torch.tensor(abs(np.transpose(ideal1[:, :256-pad_zeros]))), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.show()

    recon_spec = out[:256-pad_zeros, :]
    scio.savemat('recon_spec.mat', {'recon_spec': recon_spec})
    plt.show()



if __name__ == "__main__":
    device = torch.device("cuda:2")
    N1 = 170

    array = 10
    total_output = torch.zeros(1, N2, N1)

    ideal = np.transpose(ideal)
    ideal = torch.tensor(ideal)


    ideal1d=ideal[:, array].clone().detach()

    under = abs(torch.tensor(np.transpose(under)))


    under1d = under[:, array].clone().detach()

    net = torch.load("edhr_rate0.15_170_maskfixed.pth", map_location=device)

    net_test_2d(device)