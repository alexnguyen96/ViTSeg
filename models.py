import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from vit_pytorch import ViT
import math

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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

    def __init__(self, in_channels, out_channels, upsample_kernel=3, scale_factor=4):
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
        # pixel_shuffle: channel reduce by scale_factor**2, h&w increase by scale_factor
        x = F.pixel_shuffle(x, self.scale_factor)  
        return x


class Decoder(ViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1, scale_factor=4):
        super().__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim,
                         depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels,
                         dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.dropout_p = dropout
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width
        self.patch_num = image_size // patch_height
        num_patches = (image_size // patch_height) * (image_size // patch_width)

         # TODO: scale 4 times with scale factor 4 -> to get to 4k. Figure out the best out_channel num -> chatGPT?
        # TODO calculation to get the right img size in the end, -> find out outchanel, scalefact, kernel
        out_chan = [dim*i for i in range(2, 5)]
        self.upsampler = nn.Sequential(
          PixelShufflePack(in_channels=dim, out_channels=out_chan[0], scale_factor=scale_factor),
          PixelShufflePack(in_channels=out_chan[0], out_channels=out_chan[1], scale_factor=4),
          PixelShufflePack(in_channels=out_chan[1], out_channels=out_chan[2], scale_factor=2),
          PixelShufflePack(in_channels=out_chan[2], out_channels=num_classes, scale_factor=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # shape [4, 64, 1024]
        b, n, _ = x.shape

        # positional embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # self attention
        x = self.transformer(x)  
        # TODO: make the transformer layer deeper

        # remove the class token
        x = x[:, 1:, :]  # shape [4, 64, 1024]

        batch_num = x.shape[0]
        height = width = self.patch_num
        x = x.permute(0, 2, 1)  #shape [4, 1024, 64]
        x = x.reshape(batch_num, -1, height, width)  # shape [4, 1024, 8, 8]

        # print("x shape before upsample", x.shape)
        x = self.upsampler(x)
        # print("x shape AFTER upsample", x.shape)

        # classifying pixelwise
        x = self.softmax(x)
        x = x.argmax(dim=1)
        return x  # should be the logits


