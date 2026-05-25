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
        self.bn = torch.nn.BatchNorm1d(channel_out)


    def forward(self, inputs):
        inputs = torch.nn.functional.pad(inputs, (3*self.dilation, 3*self.dilation))
        outputs = self.dilation_conv1d(inputs)
        outputs = self.bn(outputs)

        if self.activate == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.tanh(outputs)

        return outputs


class ResnetBlock(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(ResnetBlock, self).__init__()
        self.conv_filter = Aconv1d(dilation, channel_in, channel_out, activate='tanh')
        self.conv_gate = Aconv1d(dilation, channel_in, channel_out, activate='sigmoid')

        self.conv1d = torch.nn.Conv1d(channel_out, out_channels=128, kernel_size=7,padding=3,stride=1, bias=False)
        self.bn = torch.nn.BatchNorm1d(128)

    def forward(self, inputs):
        out_filter = self.conv_filter(inputs)
        out_gate = self.conv_gate(inputs)
        outputs = out_filter * out_gate

        outputs = torch.tanh(self.bn(self.conv1d(outputs)))
        out = outputs + inputs
        return out, outputs

class WNN(torch.nn.Module):
    def __init__(self, num_classes, channels_in=1, channels_out=20, dilations=[1,2,4,8,16,32,64,128]): # dilations=[1,2,4]
        super(WNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20,kernel_size=(2,1), dilation=2**0,padding=0)
        self.relu = F.leaky_relu(negative_slope=0.2)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)
        self.conv6 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)
        self.conv7 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(2, 1), dilation=2 ** 0, padding=0)

    def forward(self, inputs):
        x = torch.tanh(self.con1d1(inputs))
        outs = 0.0
        for layer in self.resnet_block_0:
            x, out = layer(x)
            outs += out
        for layer in self.resnet_block_1:
            x, out = layer(x)
            outs += out
        for layer in self.resnet_block_2:
            x, out = layer(x)
            outs += out

        outs = torch.relu(self.bn2(self.conv1d_out(outs)))
        logits = torch.tanh(self.get_logits(outs))
        return logits