import torch
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

class Aconv1d(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(Aconv1d, self).__init__()

        assert activate in ['sigmoid', 'tanh']

        self.dilation = dilation
        self.activate = activate

        self.dilation_conv1d = torch.nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=7, dilation=self.dilation, bias=False)



    def forward(self, inputs):
        # padding number = (kernel_size - 1) * dilation / 2
        inputs = torch.nn.functional.pad(inputs, (3*self.dilation, 3*self.dilation))
        outputs = self.dilation_conv1d(inputs)
        return outputs


class ResnetBlock(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out):
        super(ResnetBlock, self).__init__()
        self.conv_relu = Aconv1d(dilation,channel_in=channel_in,channel_out=channel_out)

    def forward(self, x):

        output = self.conv_relu(x)
        outputs = torch.relu(output)
        return outputs

class MyModuleList(torch.nn.Module):
    def __init__(self, dilations=[2,4,8,16,32,64]):
        super(MyModuleList, self).__init__()
        self.layer0 = ResnetBlock(dilation=1, channel_in=1, channel_out=20)
        self.layers = torch.nn.ModuleList([

        ResnetBlock(dilation, channel_in=20, channel_out=20) for dilation in dilations])
        self.layer7 = ResnetBlock(dilation=128, channel_in=20, channel_out=20)
        self.convout=torch.nn.Conv1d(in_channels=20, out_channels=1,kernel_size=1,stride=1,bias=False)
    def forward(self, x):
        x = self.layer0(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer7(x)
        out = self.convout(x)
        return out

class WNN(torch.nn.Module):
    def __init__(self, num_classes, channels_in=1, channels_out=20, dilations=[1,2,4,8,16]): # dilations=[1,2,4]
        super(WNN, self).__init__()

        #self.dc=DCBlocks()
        self.resnet_block_0 = MyModuleList(dilations=[2,4,8,16,32,64])
        self.resnet_block_1 = MyModuleList(dilations=[2,4,8,16,32,64])
        #self.resnet_block_2 = torch.nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_2 = MyModuleList(dilations=[2,4,8,16,32,64])

    def forward(self, inputs, time_data_real, time_data_imag):
        n1 = 170
        x0 = self.resnet_block_0(inputs)
        mask = (time_data_real != 0.0).squeeze(1)

        complex = torch.view_as_complex(torch.stack([time_data_real, time_data_imag], dim=3)).squeeze(1)
        xxx = torch.fft.ifft(x0, dim=1, norm=None)
        xxx = xxx.reshape(-1, n1)

        xxx[mask] = complex[mask]
        time_pred = xxx

        sp_pred_comp = torch.fft.fft(time_pred, dim=1, norm=None)
        out0 = torch.real(sp_pred_comp).reshape(-1, 1, n1)

        x1 = self.resnet_block_1(out0)
        mask = (time_data_real != 0.0).squeeze(1)
        complex = torch.view_as_complex(torch.stack([time_data_real, time_data_imag], dim=3)).squeeze(1)
        xxx = torch.fft.ifft(x1, dim=1, norm=None)
        xxx = xxx.reshape(-1, n1)

        xxx[mask] = complex[mask]
        time_pred = xxx

        sp_pred_comp = torch.fft.fft(time_pred, dim=1, norm=None)
        out1 = torch.real(sp_pred_comp).reshape(-1, 1, n1)

        x2 = self.resnet_block_2(out1)

        return x2