import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    device = timesteps.device
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
    return embeddings    

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class AttentionUNet(torch.nn.Module):
    """
    UNet, down sampling & up sampling for global reasoning
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super(AttentionUNet, self).__init__()

        # input_channels = 3, class_number = 256, down_channel = 256

        # down_channel = kwargs['down_channel']   # default = 256
        down_channel = 64
        down_channel_1 = down_channel
        down_channel_2 = down_channel_1 * 2
        down_channel_3 = down_channel_2 * 2
        down_channel_4 = down_channel_3 * 2
        up_channel_1 = down_channel_4 * 2
        up_channel_2 = down_channel_3 * 2
        up_channel_3 = down_channel_2 * 2
        up_channel_4 = down_channel_1 * 2

        self.inc = DoubleConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel_1, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_3)
        self.down3 = DownLayer(down_channel_3, down_channel_4)
        self.down4 = DownLayer(down_channel_4, down_channel_4)
        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)
        self.up3 = UpLayer(up_channel_3, up_channel_3 // 4)
        self.up4 = UpLayer(up_channel_4, up_channel_4 // 4)
        self.outc = nn.Conv2d(up_channel_4 // 4, class_number, kernel_size=1)

        self.temb_dense0 = nn.Linear(256, 1024)
        self.temb_dense1 = nn.Linear(1024, 1024)

    def forward(self, attention_channels, t=None):
        temb = get_timestep_embedding(t, 256)
        temb = self.temb_dense0(temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense1(temb)
        
        x = attention_channels
        x0 = self.inc(x, temb)
        x1 = self.down1(x0, temb)
        x2 = self.down2(x1, temb)
        x3 = self.down3(x2, temb)
        x4 = self.down4(x3, temb)

        u1 = self.up1(x4, x3, temb)
        u2 = self.up2(u1, x2, temb)
        u3 = self.up3(u2, x1, temb)
        u4 = self.up4(u3, x0, temb)
        output = self.outc(u4)
        # attn_map as the shape of: batch_size x width x height x class
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))
        self.temb_proj = nn.Linear(1024, out_ch)


    def forward(self, x, temb):
        x = self.conv0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv1(x)
        return x


# class InConv(nn.Module):

#     def __init__(self, in_ch, out_ch):
#         super(InConv, self).__init__()
#         self.conv = DoubleConv(in_ch, out_ch)
        
        

#     def forward(self, x, timesteps=None):
#         if timesteps is not None:
#             temb = self.time_mlp(timesteps)
#             temb = temb.unsqueeze(-1).unsqueeze(-1)
#             x = x + temb
#         x = self.conv(x)
#         return x


class DownLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()

        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, temb=None):
        x = self.max_pooling(x)
        x = self.conv(x, temb)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)


    def forward(self, x1, x2, temb=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                        diffY // 2))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x, temb)
        return x


# class OutConv(nn.Module):

#     def __init__(self, in_ch, out_ch):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(in_ch),
#             nn.Linear(in_ch, in_ch),
#             nn.GELU(),
#             nn.Linear(in_ch, in_ch),
#         )

#     def forward(self, x, timesteps=None):
#         if timesteps is not None:
#             temb = self.time_mlp(timesteps)
#             temb = temb.unsqueeze(-1).unsqueeze(-1)
#             x = x + temb
#         x = self.conv(x)
#         return x