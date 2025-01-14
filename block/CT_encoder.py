import torch
import torch.nn as nn
from block.visionEmbedding import VisionEmbedding

class CT_Encoder(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=2,
        in_channels=4,
        embed_dim=1024,
        contain_mask_token=True, 
        reduction_ratio=14,
    ):
        super().__init__()

        self.vision_embedding = VisionEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            contain_mask_token=contain_mask_token, 
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((int((img_size/patch_size)**2),1))
        self.max_pool = nn.AdaptiveMaxPool2d((int((img_size/patch_size)**2),1))

        self.fc = nn.Sequential(
            nn.Linear(int((img_size/patch_size)**2), int((img_size/patch_size)**2 / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int((img_size/patch_size)**2 / reduction_ratio), int((img_size/patch_size)**2))
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, x: torch.Tensor):
        x = self.vision_embedding(x)
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        weight = (self.sigmoid(avg_out + max_out)).unsqueeze(-1)
        # print(f'weight.shape:{weight.shape}')
        x = x * weight
        x = self.norm(x)
        return weight, x