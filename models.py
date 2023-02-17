import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from vit_pytorch.vit import Attention, FeedForward
from vit_pytorch import ViT


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio. Typically, 2 to 4.
            Depends on the problem. 2 is often enough but if the problem requires more fine grain, use 4
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels, out_channels, upsample_kernel=4, scale_factor=3):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)  # channel reduce by scale_factor**2, h&w increase by scale_factor
        return x

class ViTSeg(ViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0.1, emb_dropout = 0.1):
        super(ViTSeg, self).__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim,
                                     depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels,
                                     dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.dropout_p = dropout
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        self.upsampler = PixelShufflePack(in_channels=dim, out_channels=num_classes)
        self.linear_fuse = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(dim, eps=1e-5)
        self.linear_pred = nn.Conv2d(dim, num_classes, kernel_size=(1, 1))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        #TODO: make the transformer layer deeper

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # modification starts here
        # x = self.to_latent(x)
        # x = self.mlp_head(x)

        # x = self.upsampler(x)
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout_p)
        x = self.linear_pred(x)
        return x

