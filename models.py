import torch
from torch import nn
import torch.nn.functional as F
import math

from decoder import Decoder
from encoder import SegFormerEncoder, MAEEncoder


class VitSeg(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, class_num=10,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.mae_encoder = MAEEncoder(img_size, patch_size, in_chans, embed_dim,
                                      depth, num_heads, mlp_ratio, norm_layer)

        self.seg_encoder = SegFormerEncoder()  # mit_b0 right now: (embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2))

        attn_size = math.sqrt(img_size//patch_size)
        # self.decoder = Decoder(encoder_size=attn_size, num_classes=class_num)

    def forward(self, img):
        y = self.mae_encoder(img, mask_ratio=0.75)
        print("y shape", y.shape)
        # y = self.mae_encoder.unpatchify(y)
        # y = torch.einsum('nchw->nhwc', y).detach().cpu()
        # y shape: [1, 64, 1024] == [batch, 8*8, embed_dim]

        # x = self.seg_encoder(img)
        # for layer in x:
        #     print('layer shape', layer.shape)
            # layer shape torch.Size([1, 32, 64, 64])
            # layer shape torch.Size([1, 64, 32, 32])
            # layer shape torch.Size([1, 160, 16, 16])
            # layer shape torch.Size([1, 256, 8, 8])

        # output = self.decoder(y, x)

        return y
