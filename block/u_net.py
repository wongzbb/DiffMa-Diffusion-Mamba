###  simple unet diffusion


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)

        self.up1 = Up(1280,512, bilinear)
        self.up2 = Up(640,256, bilinear)
        self.up3 = Up(320, 128, bilinear)
        self.up4 = Up(160, 64, bilinear)  
        self.up5 = Up(80, 32, bilinear)  

        self.outc = OutConv(32, out_channels)
        self.fc_c1 = nn.Linear(1024, 1280)  
        self.activation1 = nn.ReLU()
        self.fc_c2 = nn.Linear(1280, 32768)
        self.activation2 = nn.ReLU()

        
        self.t_embedder = TimestepEmbed(512)

    def forward(self, x, t, y, y2, w):
        t = self.t_embedder(t)             
        y2 = torch.mean(y2, dim=1)         
        c1 = t + y 
        c2 = t + y2
        c = torch.cat((c1, c2), dim=1)        

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        c = self.fc_c1(c) 
        c = self.activation1(c)
        c = self.fc_c2(c)
        c = self.activation2(c)


        c = c.view(c.size(0), 512, 8, 8) 
        c = F.interpolate(c, size=x6.size()[2:], mode='nearest') 
        x7 = torch.cat([x6, c], dim=1) 

        x = self.up1(x7, x5) 
        x = self.up2(x, x4) 
        x = self.up3(x, x3) 
        x = self.up4(x, x2) 
        x = self.up5(x, x1) 

        logits = self.outc(x)
        logits = F.adaptive_avg_pool2d(logits, (28, 28))
        return logits
    
    def forward_with_cfg(self, x, t, y, y2, w, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, y2, w)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

'''
unet = UNet(n_channels=4, out_channels=8, bilinear=True)
print((f"DiM Parameters: {sum(p.numel() for p in unet.parameters()):,}"))
input_images = torch.rand(1, 4, 28, 28)  
input_c = torch.rand(1, 512)  
t=torch.randint(1,100,(1,))
y2=torch.rand(1,16,512)
w=torch.rand(1,3)

output = unet(input_images, t, input_c, y2, w)
print(f"DiM Parameters: {sum(p.numel() for p in unet.parameters()):,}")
print("Output shape:", output.shape)

'''