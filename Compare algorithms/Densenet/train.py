import os
import torch
from datetime import datetime
from earlystop import EarlyStopping
from demo import *
# from wave import *
from model import *
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


epochs = 400
lr = 0.0003
bs = 32  # Batch size
train_loss = []
test_loss = []
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def net_train(device=None):
    device=device
    train_set = create_dataset(phase='train')
    train_loader = create_dataloader(
        train_set, phase='train')

    val_set = create_dataset(phase='val')
    test_loader = create_dataloader(
        val_set, phase='val')

    log_file_path = f"{model_name}_loss_log.txt"
    log_file = open(log_file_path, 'w')
    # with open('log.txt', 'w') as log_file:
    #     log_file.write('Epoch\tTrain Loss\tValidation Loss\n')
    log_file.write('Epoch\tTrain Loss\tValidation Loss\n')

    for epoch in range(epochs):
        print("迭代第{}次，当前时间：{}".format(epoch + 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        net.train()
        train_loss_list = []

        for step, (NOISE_input, GT_label, factor_matrix, xx_real, xx_imag) in enumerate(train_loader):
            NOISE_input = NOISE_input.type(torch.FloatTensor)
            GT_label = GT_label.type(torch.FloatTensor)
            factor_matrix = factor_matrix.type(torch.FloatTensor)
            xx_real = xx_real.type(torch.FloatTensor)
            xx_imag = xx_imag.type(torch.FloatTensor)

            NOISE_input = NOISE_input.to(device)
            GT_label = GT_label.to(device)
            factor_matrix = factor_matrix.to(device)
            xx_real = xx_real.to(device)
            xx_imag = xx_imag.to(device)
            output = net(NOISE_input, xx_real, xx_imag)

            loss = torch.sum(torch.abs((GT_label - output)))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            train_loss_list.append(loss.item())
            avg_train_loss = np.average(train_loss_list)
            optimizer.step()

        print("训练loss:{}".format(np.average(train_loss_list)))


        valid_loss_list = []

        net.eval()
        with torch.no_grad():
            for step, (NOISE_input, GT_label, factor_matrix, xx_real, xx_imag) in enumerate(test_loader):
                NOISE_input = NOISE_input.type(torch.FloatTensor)
                GT_label = GT_label.type(torch.FloatTensor)
                factor_matrix = factor_matrix.type(torch.FloatTensor)
                xx_real = xx_real.type(torch.FloatTensor)
                xx_imag = xx_imag.type(torch.FloatTensor)
                NOISE_input = NOISE_input.to(device)
                GT_label = GT_label.to(device)
                factor_matrix = factor_matrix.to(device)
                xx_real = xx_real.to(device)
                xx_imag = xx_imag.to(device)
                output = net(NOISE_input, xx_real, xx_imag)

                val_loss = torch.sum(torch.abs((GT_label - output)))
                valid_loss_list.append(val_loss.item())

        avg_valid_loss = np.average(valid_loss_list)
        print("验证集loss:{}".format(avg_valid_loss))
        early_stopping(avg_valid_loss, net)
        if early_stopping.early_stop:
            print("此时早停！")
            break

        # Log the loss for the current epoch
        log_file.write(f"{epoch + 1}\t{avg_train_loss:.6f}\t{avg_valid_loss:.6f}\n")

        lr = optimizer.param_groups[0]['lr']
        print("epoch={}, lr={}".format(epoch+1, lr))

if __name__ == "__main__":
    device = torch.device("cuda:2")
    net = DenseNet()

    model_name = 'dense_rate0.2_170_maskrandom_solve'
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

    net = net.to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    net_train(device)

    torch.save(net, f"{model_name}.pth")