import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class AttentionUNet(torch.nn.Module):
    """
    UNet, down sampling & up sampling for global reasoning
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super(AttentionUNet, self).__init__()

        # input_channels = 3, class_number = 256, down_channel = 256

        down_channel = kwargs['down_channel']   # default = 256

        down_channel_2 = down_channel * 2   # 512
        up_channel_1 = down_channel_2 * 2   # 1024
        up_channel_2 = down_channel * 2     # 512

        self.inc = InConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_2)

        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)
        self.outc = OutConv(up_channel_2 // 4, class_number)

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(config.hidden_size),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        # )

    def forward(self, attention_channels, timesteps=None):
        """
        Given multi-channel attention map, return the logits of every one mapping into 3-class
        :param attention_channels:
        :return:
        """
        # attention_channels as the shape of: batch_size x channel x width x height
        x = attention_channels  # [bs, 3, min_height, min_height]
        x1 = self.inc(x, timesteps)        # [bs, 256, min_height, min_height]
        x2 = self.down1(x1, timesteps)     # [bs, 512, min_height//2, min_height//2]
        x3 = self.down2(x2, timesteps)     # [bs, 512, min_height//4, min_height//4]
        x = self.up1(x3, x2, timesteps)    # [bs, 256, min_height//2, min_height//2]
        x = self.up2(x, x1, timesteps)     # [bs, 128, min_height, min_height]
        output = self.outc(x, timesteps)   # [bs, 256, min_height, min_height]
        # attn_map as the shape of: batch_size x width x height x class
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, in_ch),
        )
        

    def forward(self, x, timesteps=None):
        if timesteps is not None:
            temb = self.time_mlp(timesteps)
            temb = temb.unsqueeze(-1).unsqueeze(-1)
            x = x + temb
        x = self.conv(x)
        return x


class DownLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(in_ch),
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x, timesteps=None):
        if timesteps is not None:
            temb = self.time_mlp(timesteps)
            temb = temb.unsqueeze(-1).unsqueeze(-1)
            x = x + temb
        x = self.maxpool_conv(x)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(in_ch),
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x1, x2, timesteps=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                        diffY // 2))
        x = torch.cat([x2, x1], dim=1)

        if timesteps is not None:
            temb = self.time_mlp(timesteps)
            temb = temb.unsqueeze(-1).unsqueeze(-1)
            x = x + temb

        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(in_ch),
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x, timesteps=None):
        if timesteps is not None:
            temb = self.time_mlp(timesteps)
            temb = temb.unsqueeze(-1).unsqueeze(-1)
            x = x + temb
        x = self.conv(x)
        return x