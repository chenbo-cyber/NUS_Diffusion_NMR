import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from inspect import isfunction
from collections import deque
import copy
from types import SimpleNamespace
import scipy.io as scio

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


def clones(module, N):
    """Cloning model blocks"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DenseLayer(nn.Module):

    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(c_in=c_in + layer_idx * growth_rate,
                           # Input channels are original plus the feature maps from previous layers
                           bn_size=bn_size,
                           growth_rate=growth_rate,
                           act_fn=act_fn)
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(nn.Module):  # TODO data consistency

    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):

    def __init__(self, in_filters=8, num_layers=[4, 4, 4, 4], bn_size=2, growth_rate=8,  data_path=None, act_fn_name="relu", **kwargs):
        super().__init__()
        self.data_path = data_path
        self.hparams = SimpleNamespace(in_filters=in_filters,
                                       num_layers=num_layers,
                                       bn_size=bn_size,
                                       growth_rate=growth_rate,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.in_filters  # The start number of hidden channels
        self.conv = clones(nn.Conv2d(2, c_hidden, kernel_size=3, padding_mode='zeros', padding='same'), 5)  # TODO
        self.denseblock = clones(DenseBlock(c_in=c_hidden,
                                            num_layers=8,
                                            bn_size=self.hparams.bn_size,
                                            growth_rate=self.hparams.growth_rate,
                                            act_fn=self.hparams.act_fn), 5)
        self.transitionlayer = clones(
            TransitionLayer(c_in=c_hidden + 8 * self.hparams.growth_rate, c_out=2, act_fn=self.hparams.act_fn), 5)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, fid):
        batch_meta = None
        real_data_path = None
        if self.data_path is not None and isinstance(self.data_path, dict):
            batch_meta = self.data_path.get('batch_meta', None)
            real_data_path = self.data_path.get('real_data_path', None)

        slice_runtime = []
        if batch_meta:
            for raw_meta in batch_meta:
                start = int(raw_meta['start'])
                end = int(raw_meta['end'])
                pad_shape = tuple(raw_meta['pad_shape'])
                pad_height = int(raw_meta['pad_height'])
                pad_width = int(raw_meta['pad_width'])
                fid_full = reconstruct_data_ddpm(fid[start:end].squeeze(1), pad_shape, pad_height, pad_width)
                fid_mask = torch.ne(fid_full, 0)
                slice_runtime.append({
                    'start': start,
                    'end': end,
                    'pad_shape': pad_shape,
                    'pad_height': pad_height,
                    'pad_width': pad_width,
                    'fid_full': fid_full,
                    'fid_mask': fid_mask
                })
        elif real_data_path is not None:
            ori_spec = scio.loadmat(f"{real_data_path}.mat")['spec']
            _, pad_shape, pad_height, pad_width = block_data_ddpm(ori_spec, 64)
            fid_full = reconstruct_data_ddpm(fid.squeeze(1), pad_shape, pad_height, pad_width)
            fid_mask = torch.ne(fid_full, 0)
            slice_runtime.append({
                'start': 0,
                'end': int(fid.shape[0]),
                'pad_shape': tuple(pad_shape),
                'pad_height': int(pad_height),
                'pad_width': int(pad_width),
                'fid_full': fid_full,
                'fid_mask': fid_mask
            })

        y = fid
        for block_idx, num_layers in enumerate(self.hparams.num_layers):
            x = self.conv[block_idx](x)
            x = self.denseblock[block_idx](x)
            x = self.transitionlayer[block_idx](x)
            if slice_runtime:
                x_complex = torch.complex(x[:, 0], x[:, 1])
                updated_block_list = []
                for runtime_meta in slice_runtime:
                    start = runtime_meta['start']
                    end = runtime_meta['end']
                    pad_shape = runtime_meta['pad_shape']
                    pad_height = runtime_meta['pad_height']
                    pad_width = runtime_meta['pad_width']
                    fid_full = runtime_meta['fid_full']
                    fid_mask = runtime_meta['fid_mask']

                    x_slice = reconstruct_data_ddpm(x_complex[start:end], pad_shape, pad_height, pad_width)
                    x_slice = torch.fft.ifft2(x_slice, dim=(-2, -1))
                    x_slice[fid_mask] = (
                        1 * x_slice[fid_mask] + 1e3 * fid_full[fid_mask]
                    ) / (1 + 1e3)
                    x_slice = torch.fft.fft2(x_slice, dim=(-2, -1))
                    x_blocks, _, _, _ = block_data_ddpm(x_slice, 64)
                    updated_block_list.append(x_blocks.unsqueeze(1))
                x = torch.cat(updated_block_list, dim=0)
            else:
                x = torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)
                x = torch.fft.ifft2(x, dim=(-2, -1))
                y_mask = torch.ne(y, 0)
                x[y_mask] = (x[y_mask] + 1e3 * y[y_mask]) / (1 + 1e3)
                x = torch.fft.fft2(x, dim=(-2, -1))


            x = torch.concat((x.real, x.imag), dim=1)
            # Z.append(x)
        # Z = torch.cat(Z, dim=1)
        return x

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SpecialConcat(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 2), stride=1, padding=0)

    def forward(self, x, feats):
        if x.shape[-1] != feats.shape[-1]:
            x = self.conv(x)
            # x = F.interpolate(x, size=feats.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.cat((x, feats), dim=1)
        # print("concat shape: ", out.shape)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding



class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class SelfAttention2varible(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm_q = nn.GroupNorm(norm_groups, in_channel)
        self.norm_kv = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input_q, input_kv):
        batch, channel, height, width = input_q.shape
        n_head = self.n_head
        head_dim = channel // n_head


        norm_q = self.norm_q(input_q)
        query = self.q(norm_q).view(batch, n_head, head_dim, height, width)


        norm_kv = self.norm_kv(input_kv)
        kv = self.kv(norm_kv).view(batch, n_head, head_dim * 2, height, width)
        key, value = kv.chunk(2, dim=2)


        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)


        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))


        return out + input_q

class SelfAttention3varible(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm_q = nn.GroupNorm(norm_groups, in_channel)
        self.norm_k = nn.GroupNorm(norm_groups, in_channel)
        self.norm_v = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.k = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.v = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input_q, input_k, input_v):
        batch, channel, height, width = input_q.shape
        n_head = self.n_head
        head_dim = channel // n_head


        norm_q = self.norm_q(input_q)
        query = self.q(norm_q).view(batch, n_head, head_dim, height, width)


        norm_k = self.norm_k(input_k)
        key = self.k(norm_k).view(batch, n_head, head_dim, height, width)


        norm_v = self.norm_v(input_v)
        value = self.v(norm_v).view(batch, n_head, head_dim, height, width)


        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)


        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))


        return out + input_q


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x)
        return x

class ResnetBlocWith21Attn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention2varible(dim_out, norm_groups=norm_groups)

    def forward(self, x, x_sr, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x, x_sr)
        return x

class ResnetBlocWith22Attn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention2varible(dim_out, norm_groups=norm_groups)

    def forward(self, x, x_sr, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x, x_sr)
        return x


class DPM(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super(DPM, self).__init__()
        self.conv_head = nn.Conv2d(in_channels, out_channels, 1)

        self.conv3_1_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.cca = CCALayer(out_channels)
        self.depthwise = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels)
        self.depthwise_dilated = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=6, groups=out_channels,
                                           dilation=3)
        self.conv_tail = nn.Conv2d(out_channels, in_channels, 1)
        self.active = nn.Sigmoid()

    def forward(self, input):
        input_h = self.conv_head(input)
        x = self.conv3_1_A(input_h) + self.conv3_1_B(input_h)

        x_cca = self.cca(x)
        x_de = self.depthwise(x_cca + input_h)
        x_de = self.depthwise_dilated(x_de)
        x_de = x_de + x_cca
        x_fea = self.active(self.conv_tail(x_de))
        return (x_fea * input)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    eps = 1e-7
    F_variance = ((F - F_mean + eps).pow(2)).sum(dim=(2, 3), keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(dim=(2, 3), keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            data_path=None,
            with_noise_level_emb=True,
            image_size=64
    ):
        super().__init__()
        self.data_path = data_path

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWith21Attn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                 norm_groups=norm_groups,
                                 dropout=dropout, with_attn=True),
            ResnetBlocWith22Attn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                 norm_groups=norm_groups,
                                 dropout=dropout, with_attn=True),
        ])

        ups = []
        cats = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                cats.append(SpecialConcat(pre_channel))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                cats.append(SpecialConcat(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.cats = nn.ModuleList(cats)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        self.dpm = DPM(in_channel, inner_channel)
        self.adapt_conv = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1)

        self.densenet = DenseNet(data_path=data_path)


    def forward(self, fid, x_sr, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        def generate_noise(snr, max_val, N1, N2, device):
            snr = snr.to(device)
            max_val = max_val.to(device)
            std_noise = max_val / (2 * snr)
            noise_real = torch.normal(mean=torch.zeros((N1, N2), device=device), std=std_noise)
            noise_imag = torch.normal(mean=torch.zeros((N1, N2), device=device), std=std_noise)
            noise_out = noise_real + 1j * noise_imag
            return noise_out
        x_sr_complex = torch.complex(x_sr[:, 0], x_sr[:, 1])
        x_sr_noise = x_sr_complex + generate_noise((torch.rand(1) * (90 - 30) + 30).float(), torch.max(torch.abs(x_sr_complex)), 64, 64, device=x_sr_complex.device)
        x_sr_noise = x_sr_noise.unsqueeze(1)
        x_sr_noise = torch.concat((x_sr_noise.real, x_sr_noise.imag), dim=1)
        x_srdc = self.densenet(x_sr_noise, fid)
        x_den = x_srdc
        # x_sr = self.adapt_conv(x_sr)
        x_sr = self.dpm(x_sr)

        outputs = deque()
        first_layer = True
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x_sr = layer(x_sr, t)
            else:
                x_sr = layer(x_sr)
            if first_layer or x_sr.shape[2] != outputs[-1].shape[2]:
                outputs.append(x_sr)
                first_layer = False

        outputs_dc = deque()
        first_layer = True
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x_srdc = layer(x_srdc, t)
            else:
                x_srdc = layer(x_srdc)
            if first_layer or x_srdc.shape[2] != outputs_dc[-1].shape[2]:
                outputs_dc.append(x_srdc)
                first_layer = False

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                if outputs and outputs[0].shape[2] == x.shape[2]:
                    x = layer(x + outputs.popleft(), t)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWith21Attn):
                x = layer(x, x_sr, t)
            elif isinstance(layer, ResnetBlocWith22Attn):
                x = layer(x, x_srdc, t)
            else:
                x = layer(x)

        for layer, cat_layer in zip(self.ups, self.cats):
            if isinstance(layer, ResnetBlocWithAttn):
                if outputs_dc and outputs_dc[-1].shape[2] == feats[-1].shape[2] and outputs_dc[-1].shape[1] == feats[-1].shape[1]:
                    x = layer(cat_layer(x, feats.pop() + outputs_dc.pop()), t)
                else:
                    x = layer(cat_layer(x, feats.pop()), t)
            else:
                x = layer(x)

        return self.final_conv(x), x_den

def pad_data_ddpm(data, block_size):

    height, width = data.shape


    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size


    if isinstance(data, torch.Tensor):
        padded_data = F.pad(data, (0, pad_width, 0, pad_height), mode='constant', value=0)
    elif isinstance(data, np.ndarray):
        padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_data, pad_height, pad_width

def block_data_ddpm(data, block_size):

    padded_data, pad_height, pad_width = pad_data_ddpm(data, block_size)


    height, width = padded_data.shape


    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size


    if isinstance(data, torch.Tensor):
        blocks = torch.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype).to(data.device)
    elif isinstance(data, np.ndarray):
        blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype)


    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[idx] = block
            idx += 1

    return blocks, padded_data.shape, pad_height, pad_width

def reconstruct_data_ddpm(blocks, pad_shape, pad_height, pad_width):

    height, width = pad_shape


    num_blocks, block_height, block_width = blocks.shape


    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width


    reconstructed_data = torch.empty((height, width), dtype=blocks.dtype).to(blocks.device)


    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i * block_height:(i + 1) * block_height,j * block_width:(j + 1) * block_width] = block
            idx += 1


    final_data = reconstructed_data[:height if pad_height == 0 else -pad_height, :width if pad_width == 0 else -pad_width]

    return final_data