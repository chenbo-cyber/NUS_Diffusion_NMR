import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from datetime import datetime
from earlystop import EarlyStopping
from demo_dc import *
from WNN1 import WNN
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import scipy.io as sio


epochs = 400
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"


parser = ArgumentParser(description='ISTA-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=120, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=24, help='phase number of ISTA-Net')
parser.add_argument('--learning_rate', type=float, default=0.00003, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=15, help='from {1, 4, 10, 15, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='4', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
bs = 32

num = 40000

def net_train(device=None):
    device = device
    train_set = create_dataset(phase='train')
    train_loader = create_dataloader(
        train_set, phase='train')

    val_set = create_dataset(phase='val')
    test_loader = create_dataloader(
        val_set, phase='val')


    log_file_path = f"{model_name}_loss_log.txt"
    log_file = open(log_file_path, 'w')
    log_file.write('Epoch\tTrain Loss\tValidation Loss\n')


    for epoch in range(epochs):
        print("迭代第{}次，当前时间：{}".format(epoch + 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        net.train()
        train_loss_list = []

        for step, (NOISE_input, GT_label, xx_real, xx_imag) in enumerate(train_loader):
            NOISE_input = NOISE_input.type(torch.FloatTensor)
            GT_label = GT_label.type(torch.FloatTensor)
            NOISE_input = NOISE_input.to(device)
            GT_label = GT_label.to(device)

            xx_real = xx_real.type(torch.FloatTensor)
            xx_real = xx_real.to(device)

            xx_imag = xx_imag.type(torch.FloatTensor)
            xx_imag = xx_imag.to(device)

            output = net(NOISE_input, xx_real, xx_imag)


            def CDMANE(output, label):
                a = 0.5
                cdmane = torch.sum(torch.abs((label - output) / (torch.abs(label) + a)))
                return cdmane

            loss = torch.sum(torch.abs((GT_label - output)))

            optimizer.zero_grad()
            loss.backward()

            train_loss_list.append(loss.item())
            avg_train_loss = np.average(train_loss_list)
            optimizer.step()

        print("训练loss:{}".format(np.average(train_loss_list)))

        valid_loss_list = []

        net.eval()
        with torch.no_grad():
            for step, (NOISE_input, GT_label, xx_real, xx_imag) in enumerate(test_loader):
                NOISE_input = NOISE_input.type(torch.FloatTensor)
                GT_label = GT_label.type(torch.FloatTensor)
                NOISE_input = NOISE_input.to(device)
                GT_label = GT_label.to(device)

                xx_real = xx_real.type(torch.FloatTensor)
                xx_real = xx_real.to(device)

                xx_imag = xx_imag.type(torch.FloatTensor)
                xx_imag = xx_imag.to(device)
                output = net(NOISE_input, xx_real, xx_imag)

                def CDMANE(output, label):
                    a = 0.5
                    cdmane = torch.sum(torch.abs((label - output) / (torch.abs(label) + a)))
                    return cdmane
                val_loss = torch.sum(torch.abs((GT_label - output)))
                valid_loss_list.append(val_loss.item())

        avg_valid_loss = np.average(valid_loss_list)
        print("验证集loss:{}".format(avg_valid_loss))
        early_stopping(avg_valid_loss, net)
        if early_stopping.early_stop:
            print("此时早停！")
            break
        log_file.write(f"{epoch + 1}\t{avg_train_loss:.6f}\t{avg_valid_loss:.6f}\n")

        lr = optimizer.param_groups[0]['lr']
        print("epoch={}, lr={}".format(epoch + 1, lr))



if __name__ == "__main__":
    device = torch.device("cuda:1")
    net = WNN(1, 1)#.to(device0)

    model_name = 'WNN_0.2_170_maskrandom'

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    net_train(device)

    torch.save(net, f"{model_name}.pth")