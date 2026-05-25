import torch
import torch.nn as nn



class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv1d, self).__init__()

        # 深度卷积：每个输入通道单独进行卷积
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)

        # 逐点卷积：通过1x1卷积对深度卷积的输出进行线性组合
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)  # 应用深度卷积
        x = self.pointwise(x)  # 应用逐点卷积
        return x

class Input(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Input, self).__init__()
        # self.conv1 = snn.SeparableConv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1 = SeparableConv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        return out

class upper_EDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upper_EDBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        # self.conv1sep = snn.SeparableConv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1sep = SeparableConv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2trans = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)
        # self.conv3sep = snn.SeparableConv1d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3sep = SeparableConv1d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU()
    def forward(self,x):
        x1 = x
        out1 = self.bn1(x)
        out2 = self.conv1sep(out1)
        out3 = self.relu1(out2)
        out4 = self.bn2(out3)
        out5 = self.conv2trans(out4)
        out6 = self.relu2(out5)
        out7 = torch.cat((x1, out6), dim=1)
        out8 = self.bn3(out7)
        out9 = self.conv3sep(out8)
        out10 = self.relu3(out9)
        return out10



class down_EDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_EDBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(128)
        # self.conv1sep = snn.SeparableConv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv1sep = SeparableConv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2trans = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(256)
        # self.conv3sep = snn.SeparableConv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv3sep = SeparableConv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU()
    def forward(self, x):
        x1 = x
        out1 = self.bn1(x)
        out2 = self.conv1sep(out1)
        out3 = self.relu1(out2)
        out4 = self.bn2(out3)
        out5 = self.conv2trans(out4)
        out6 = self.relu2(out5)
        out7 = torch.cat((x1, out6), dim=1)
        out8 = self.bn3(out7)
        out9 = self.conv3sep(out8)
        out10 = self.relu3(out9)
        return out10


class hight_to_lowblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hight_to_lowblock, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.downconv = nn.Conv1d(in_channels=64, out_channels=128,kernel_size=5, stride=2, padding=0)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        out = self.bn1(x)
        out = self.downconv(out)
        out = self.relu(out)
        return out

class low_to_hightblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(low_to_hightblock, self).__init__()
        self.bn1 = nn.BatchNorm1d(128)
        self.downconv = nn.ConvTranspose1d(in_channels=128, out_channels=64,kernel_size=6, stride=2, padding=0)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        out = self.bn1(x)
        out = self.downconv(out)
        out = self.relu(out)
        return out

class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output,self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        out = self.conv(x)
        return out


class EDHRnet(nn.Module):
    def __init__(self):
        super(EDHRnet, self).__init__()
        self.input = Input(1, 64)
        self.edblock1upper = upper_EDBlock(64, 64)
        self.h_to_l1 = hight_to_lowblock(64, 128)
        self.edblock2upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.edblock3upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.h_to_l3 = hight_to_lowblock(64, 128)
        self.l_to_h3 = low_to_hightblock(128, 64)
        self.edblock4upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.edblock5upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.h_to_l5 = hight_to_lowblock(64, 128)
        self.l_to_h5 = low_to_hightblock(128, 64)
        self.edblock6upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.edblock7upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.edblock8upper = upper_EDBlock(in_channels=64, out_channels=64)
        self.edblock2down = down_EDBlock(in_channels=128, out_channels=128)
        self.edblock3down = down_EDBlock(in_channels=128, out_channels=128)
        self.edblock4down = down_EDBlock(in_channels=128, out_channels=128)
        self.edblock5down = down_EDBlock(in_channels=128, out_channels=128)
        self.edblock6down = down_EDBlock(in_channels=128, out_channels=128)
        self.edblock7down = down_EDBlock(in_channels=128, out_channels=128)
        self.l_to_h7 = low_to_hightblock(128, 64)
        self.out = Output(64, 1)
    def forward(self, x):
        x1 = self.input(x)
        x1_hight = self.edblock1upper(x1)
        x1_d = self.h_to_l1(x1_hight)
        x2_hight = self.edblock2upper(x1_hight)
        x3_hight = self.edblock3upper(x2_hight)
        x2_low = self.edblock2down(x1_d)
        x3_low = self.edblock3down(x2_low)
        x3_u =self.l_to_h3(x3_low)
        x3_d = self.h_to_l3(x3_hight)
        x3_hight1 = (x3_hight + x3_u)
        x3_low1 = (x3_low + x3_d)

        x4_hight = self.edblock4upper(x3_hight1)
        x5_hight = self.edblock5upper(x4_hight)
        x5_d = self.h_to_l5(x5_hight)
        x4_low = self.edblock4down(x3_low1)
        x5_low = self.edblock5down(x4_low)
        x5_u = self.l_to_h5(x5_low)
        x5_hight1 = (x5_hight + x5_u)
        x5_low1 = (x5_low + x5_d)

        x6_hight = self.edblock6upper (x5_hight1)
        x7_hight = self.edblock7upper (x6_hight)
        x6_low = self.edblock6down(x5_low1)
        x7_low = self.edblock7down(x6_low)
        x7_u = self.l_to_h7(x7_low)
        x7_hight1 = (x7_hight + x7_u)
        x8_hight = self.edblock8upper(x7_hight1)
        out = self.out(x8_hight)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)